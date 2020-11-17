//
// Created by wyz on 20-11-6.
//
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/device_new.h>
#include "bvh.h"
#include<chrono>
#include<iostream>
#include<algorithm>
#include<helper_cuda.h>
#include<queue>
#include<unordered_map>
class CuBVH_Impl{
public:
    CuBVH_Impl():d_tris_data_(nullptr){}
    ~CuBVH_Impl();
public:
    Triangle* h_tris_data_,*d_tris_data_;
    std::vector<std::pair<Triangle*,Triangle*>> tris_;
    BVHNode* CuRecursiveBuild(std::vector<std::pair<Triangle*,Triangle*>>& tris);
    BVHNode* root_;
    BVHNode* h_nodes_,*d_nodes_;
    queue<BVHNode*> h_nodes_q_;
    unordered_map<BVHNode*,BVHNode*> h2d;
};
CuBVH_Impl::~CuBVH_Impl()
{
    if(h_tris_data_) delete[] h_tris_data_;
    if(h_nodes_) delete[] h_nodes_;
    if(d_tris_data_) cudaFree(d_tris_data_);
    if(d_nodes_) cudaFree(d_nodes_);
}
Triangle *BVH::GetDeviceTris() const
{
    return impl_->d_tris_data_;
}
Triangle* BVH::GetHostTris() const
{
    return impl_->h_tris_data_;
}
BVHNode *BVH::GetDeviceBVHRoot() const
{
    return impl_->root_;
}
uint32_t BVH::GetDeviceTriNum() const {
    return tris_num_;
}

BVH::BVH(mesh *m, set<int>& cd_res,uint32_t max_prims_in_node,BVH::DeviceType device_type,BVH::SplitMethod split_method)
:mesh_(m),cd_res_(cd_res),max_prims_in_node_(max_prims_in_node),device_type_(device_type),split_method_(split_method),
tris_num_(m->getNbFaces())
{
    if(device_type==DeviceType::GPU){
        impl_=nullptr;
        CuBVHBuild();
        return;
    }
    std::cout<<"CPU BVH:"<<std::endl;
    auto start=std::chrono::steady_clock::now();

    assert(m!=nullptr);

    for(size_t i=0;i<tris_num_;i++){
        tri3f idx=m->_tris[i];
        Triangle* triangle=new Triangle(m->_vtxs[idx.id0()],
                                        m->_vtxs[idx.id1()],
                                        m->_vtxs[idx.id2()]);
        triangle->id=i;
        triangle->v_id_=idx;
        tris_.push_back(triangle);
    }

    root_=RecursiveBuild(tris_);
    auto end=std::chrono::steady_clock::now();
    uint32_t t=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout<<"\tCPU BVH build time taken: "<<t<<"ms"<<std::endl;

}
void BVH::CuBVHBuild()
{

    auto start=std::chrono::steady_clock::now();
    impl_=new CuBVH_Impl();
    assert(mesh_!=nullptr);

    std::cout<<"GPU BVH:"<<std::endl;

    impl_->h_tris_data_=new Triangle[tris_num_];

    //transfer ordinary tri3f and vec3f into Triangle
    for(size_t i=0;i<tris_num_;i++){
        tri3f idx=mesh_->_tris[i];
        auto& tri=impl_->h_tris_data_[i];
        tri.SetVertex(mesh_->_vtxs[idx.id0()],
                      mesh_->_vtxs[idx.id1()],
                      mesh_->_vtxs[idx.id2()]);
        tri.id=i;
        tri.v_id_=idx;
    }

    //gpu分配tris_num个Triangle对象大小的空间，不是指针！
    //load Triangle data from cpu memory upto gpu memory
    checkCudaErrors(cudaMalloc((void**)&impl_->d_tris_data_,sizeof(Triangle)*tris_num_));
    checkCudaErrors(cudaMemcpy(impl_->d_tris_data_,impl_->h_tris_data_,
                               sizeof(Triangle)*tris_num_,cudaMemcpyHostToDevice));

    for(size_t i=0;i<tris_num_;i++){
        impl_->tris_.push_back(make_pair(&impl_->h_tris_data_[i],&impl_->d_tris_data_[i]));
    }

    impl_->h_nodes_=new BVHNode[2*tris_num_-1];
    checkCudaErrors(cudaMalloc((void**)&impl_->d_nodes_,sizeof(BVHNode)*(2*tris_num_-1)));
    for(size_t i=0;i<2*tris_num_-1;i++){
        auto ptr=&impl_->h_nodes_[i];
        impl_->h_nodes_q_.push(ptr);
        impl_->h2d[ptr]=&impl_->d_nodes_[i];
    }

    impl_->root_=impl_->CuRecursiveBuild(impl_->tris_);
    if(impl_->h_nodes_q_.size()!=0)
        std::cout<<"ERROR! Queue size not equal zero, size is: "<<impl_->h_nodes_q_.size()<<std::endl;

    //将每个节点中left和right的host指针换为相应的device指针
    for(size_t i=0;i<2*tris_num_-1;i++){
        impl_->h_nodes_[i].left_=impl_->h2d[impl_->h_nodes_[i].left_];
        impl_->h_nodes_[i].right_=impl_->h2d[impl_->h_nodes_[i].right_];
    }
    impl_->root_=impl_->h2d[impl_->root_];//最后替换根节点
    checkCudaErrors(cudaMemcpy(impl_->d_nodes_,impl_->h_nodes_,sizeof(BVHNode)*(2*tris_num_-1),cudaMemcpyHostToDevice));
    auto end=std::chrono::steady_clock::now();
    uint32_t t=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout<<"\tGPU BVH build time taken: "<<t<<"ms"<<std::endl;
}

BVHNode* CuBVH_Impl::CuRecursiveBuild(std::vector<std::pair<Triangle*,Triangle*>>& tris)
{
    if(h_nodes_q_.size()==0){
        std::cout<<"ERROR! Queue is empty!"<<std::endl;
    }
    auto h_node=h_nodes_q_.front();
    h_nodes_q_.pop();

    Bound3 bound;
    for(size_t i=0;i<tris.size();i++){
        bound=bound.Union(tris[i].first->GetBound3());
    }
    if(tris.size()==1){
        h_node->bound_=tris[0].first->GetBound3();
        h_node->left_= nullptr;
        h_node->right_= nullptr;
        h_node->object_=tris[0].second;
        return h_node;
    }
    else if(tris.size()==2){
        std::vector<std::pair<Triangle*,Triangle*>> l,r;
        l.push_back(tris[0]);
        r.push_back(tris[1]);
        h_node->left_=CuRecursiveBuild(l);
        h_node->right_=CuRecursiveBuild(r);
        h_node->bound_=Union(h_node->left_->bound_,h_node->right_->bound_);
        return h_node;
    }
    else{
        Bound3 centro_id_bound;
        for(std::size_t i=0;i<tris.size();i++)
            centro_id_bound=Union(centro_id_bound,tris[i].first->GetBound3().CentroID());
        int dim=centro_id_bound.MaxExtent();
        switch (dim) {
            case 0:
                std::sort(tris.begin(),tris.end(),[](auto t1,auto t2){
                    return t1.first->GetBound3().CentroID().x<
                           t2.first->GetBound3().CentroID().x;
                });break;
            case 1:
                std::sort(tris.begin(),tris.end(),[](auto t1,auto t2){
                    return t1.first->GetBound3().CentroID().y<
                           t2.first->GetBound3().CentroID().y;
                });break;
            case 2:
                std::sort(tris.begin(),tris.end(),[](auto t1,auto t2){
                    return t1.first->GetBound3().CentroID().z<
                           t2.first->GetBound3().CentroID().z;
                });break;
        }

        auto begining=tris.begin();
        auto middling=tris.begin()+tris.size()/2;
        auto ending=tris.end();

        auto left_shapes=std::vector<std::pair<Triangle*,Triangle*>>(begining,middling);
        auto right_shapes=std::vector<std::pair<Triangle*,Triangle*>>(middling,ending);
        assert(tris.size()==(left_shapes.size()+right_shapes.size()));

        h_node->left_=CuRecursiveBuild(left_shapes);
        h_node->right_=CuRecursiveBuild(right_shapes);
        h_node->bound_=Union(h_node->left_->bound_,h_node->right_->bound_);
    }
    return h_node;
}



BVHNode *BVH::RecursiveBuild(std::vector<Triangle*>& tris)
{
    BVHNode* node=new BVHNode();

    Bound3 bound;
    for(size_t i=0;i<tris.size();i++){
        bound=bound.Union(tris[i]->GetBound3());
    }
    if(tris.size()==1){
        node->bound_=tris[0]->GetBound3();
        node->left_= nullptr;
        node->right_= nullptr;
        node->object_=tris[0];
        return node;
    }
    else if(tris.size()==2){
        std::vector<Triangle*> l,r;
        l.push_back(tris[0]);
        r.push_back(tris[1]);
        node->left_=RecursiveBuild(l);
        node->right_=RecursiveBuild(r);
        node->bound_=Union(node->left_->bound_,node->right_->bound_);
        return node;
    }
    else{
        Bound3 centro_id_bound;
        for(std::size_t i=0;i<tris.size();i++)
            centro_id_bound=Union(centro_id_bound,tris[i]->GetBound3().CentroID());
        int dim=centro_id_bound.MaxExtent();
        switch (dim) {
            case 0:
                std::sort(tris.begin(),tris.end(),[](auto t1,auto t2){
                    return t1->GetBound3().CentroID().x<
                           t2->GetBound3().CentroID().x;
                });break;
            case 1:
                std::sort(tris.begin(),tris.end(),[](auto t1,auto t2){
                    return t1->GetBound3().CentroID().y<
                           t2->GetBound3().CentroID().y;
                });break;
            case 2:
                std::sort(tris.begin(),tris.end(),[](auto t1,auto t2){
                    return t1->GetBound3().CentroID().z<
                           t2->GetBound3().CentroID().z;
                });break;
        }


        auto begining=tris.begin();
        auto middling=tris.begin()+tris.size()/2;
        auto ending=tris.end();

        auto left_shapes=std::vector<Triangle*>(begining,middling);
        auto right_shapes=std::vector<Triangle*>(middling,ending);
        assert(tris.size()==(left_shapes.size()+right_shapes.size()));

        node->left_=RecursiveBuild(left_shapes);
        node->right_=RecursiveBuild(right_shapes);
        node->bound_=Union(node->left_->bound_,node->right_->bound_);
    }
    return node;
}
uint32_t tmp_cd_num;
extern bool tri_contact (const vec3f &P1,const vec3f &P2,const vec3f &P3,const vec3f &Q1,const vec3f &Q2,const vec3f &Q3);
bool BVH::RecursiveCD(const Triangle *tri, const BVHNode *node)
{
    if(!node) return false;
    //非叶节点，先判断三角形的包围盒与当前节点的包围盒是否相交，相交则递归判断子节点，不想交就直接返回false
    //叶节点，使用三角形算法判断是否相交
    if(!node->IsLeafNode()){
        auto b1=tri->GetBound3();
        auto b2=node->GetBound3();
        if(Insect(b1,b2)){
            bool r1=RecursiveCD(tri,node->left_);
            bool r2=RecursiveCD(tri,node->right_);
            return r1||r2;
        }
        else
            return false;
    }
    else{
        if(node->object_->GetID()<=tri->GetID())
            return false;
        auto v_id0=tri->GetTriID();
        auto v_id1=node->object_->GetTriID();
        //判断是否同一个节点或者相临的两个三角形
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                if(v_id0.id(i)==v_id1.id(j))
                    return false;
            }
        }
        if(tri_contact(tri->V0(),tri->V1(),tri->V2(),node->object_->V0(),node->object_->V1(),node->object_->V2())){
            cd_res_.insert(tri->GetID());
            cd_res_.insert(node->object_->GetID());
            tmp_cd_num++;
//            auto id1=tri->GetID();
//            auto id2=node->object_->GetID();
//            printf("triangle contact found at (%d, %d) = (%d, %d, %d) (%d, %d, %d)\n",
//                   id1,id2,
//                   tris_[id1]->GetTriID().id(0),tris_[id1]->GetTriID().id(1),tris_[id1]->GetTriID().id(2),
//                   tris_[id2]->GetTriID().id(0),tris_[id2]->GetTriID().id(1),tris_[id2]->GetTriID().id(2));
            return true;
        }
        else return false;
    }

}

void BVH::TriangleSelfCD()
{
    auto start=std::chrono::steady_clock::now();
    uint32_t total_cd_num=0;
//    std::sort(tris_.begin(),tris_.end(),[](auto t1,auto t2){
//        return t1->GetID()<t2->GetID();
//    });
    for (std::size_t i = 0; i < tris_num_; ++i){
        tmp_cd_num=0;
        RecursiveCD(tris_[i],root_);
        if(tmp_cd_num){
            //std::cout<<"triangle id is: "<<i<<"\tcd number is: "<<tmp_cd_num<<std::endl;
        }
        total_cd_num+=tmp_cd_num;
    }
    std::cout<<"\tCPU BVH collision detected number is: "<<total_cd_num<<std::endl;
    auto end=std::chrono::steady_clock::now();
    uint32_t t=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    std::cout<<"\tCPU BVH collision detection time taken: "<<t<<"ms"<<std::endl;

}

BVH::~BVH()
{
    if(device_type_==DeviceType::CPU){
        for(size_t i=0;i<tris_num_;i++)
            delete tris_[i];
        RecursiveFree(root_);
    }
    else if(device_type_==DeviceType::GPU){
        delete impl_;
    }
}

void BVH::RecursiveFree(BVHNode* node)
{
    if(node!=nullptr){
        RecursiveFree(node->left_);
        RecursiveFree(node->right_);
        delete node;
    }
}









