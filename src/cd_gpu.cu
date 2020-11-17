//
// Created by wyz on 20-11-2.
//
#include"cd_gpu.cuh"
uint32_t blocks_per_grid;
const uint32_t threads_per_block=32;
const uint32_t MAX_CONTACT_NUM=threads_per_block-1;

void CollisionDetect::PrepareGPUResultData(uint32_t tri_num)
{
    checkCudaErrors(cudaMalloc((void**)&d_res,sizeof(uint)*tri_num*threads_per_block));
    h_res=(uint*)malloc(tri_num*sizeof(uint)*threads_per_block);
}
void CollisionDetect::PrepareGPUData()
{
    checkCudaErrors(cudaMalloc((void**)&d_obj_a_tris_,obj_a_tri_num_*sizeof(tri3f )));
    checkCudaErrors(cudaMalloc((void**)&d_obj_a_vtxs_,obj_a_vtx_num_*sizeof(vec3f )));

    checkCudaErrors(cudaMemcpy(d_obj_a_tris_,obj_a_tris_,obj_a_tri_num_*sizeof(tri3f),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_obj_a_vtxs_,obj_a_vtxs_,obj_a_vtx_num_*sizeof(vec3f),cudaMemcpyHostToDevice));

    if(!is_self_cd){
        //prepare data for obj b
        checkCudaErrors(cudaMalloc((void**)&d_obj_b_tris_,obj_b_tri_num_*sizeof(tri3f)));
        checkCudaErrors(cudaMalloc((void**)&d_obj_b_vtxs_,obj_b_vtx_num_*sizeof(vec3f )));

        checkCudaErrors(cudaMemcpy(d_obj_b_tris_,obj_b_tris_,obj_b_tri_num_*sizeof(tri3f),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_obj_b_vtxs_,obj_b_vtxs_,obj_b_vtx_num_*sizeof(vec3f),cudaMemcpyHostToDevice));
    }

    PrepareGPUResultData(obj_a_tri_num_);

}
void CollisionDetect::GenerateTrisData()
{
    std::cout<<"GenerateTrisData"<<std::endl;
    obj_a_data_=(vec3f*)malloc(obj_a_tri_num_*sizeof(vec3f)*3);
    for(uint32_t i=0;i<obj_a_tri_num_;i++){
        obj_a_data_[i*3+0]=obj_a_vtxs_[obj_a_tris_[i].id0()];
        obj_a_data_[i*3+1]=obj_a_vtxs_[obj_a_tris_[i].id1()];
        obj_a_data_[i*3+2]=obj_a_vtxs_[obj_a_tris_[i].id2()];
    }
    checkCudaErrors(cudaMalloc((void**)&d_obj_a_data_,obj_a_tri_num_*sizeof(vec3f)*3));
    checkCudaErrors(cudaMemcpy(d_obj_a_data_,obj_a_data_,obj_a_tri_num_*sizeof(vec3f)*3,cudaMemcpyHostToDevice));
    std::cout<<"obj a data generate finish..."<<std::endl;

    if(!is_self_cd){
        obj_b_data_=(vec3f*)malloc(obj_b_tri_num_*sizeof(vec3f)*3);
        for(uint32_t i=0;i<obj_b_tri_num_;i++){
            obj_b_data_[i*3+0]=obj_b_vtxs_[obj_b_tris_[i].id0()];
            obj_b_data_[i*3+1]=obj_b_vtxs_[obj_b_tris_[i].id1()];
            obj_b_data_[i*3+2]=obj_b_vtxs_[obj_b_tris_[i].id2()];
        }
        checkCudaErrors(cudaMalloc((void**)&d_obj_b_data_,obj_b_tri_num_*sizeof(vec3f)*3));
        checkCudaErrors(cudaMemcpy(d_obj_b_data_,obj_b_data_,obj_b_tri_num_*sizeof(vec3f)*3,cudaMemcpyHostToDevice));
        std::cout<<"obj b data generate finish..."<<std::endl;
    }

    checkCudaErrors(cudaMalloc((void**)&d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block));
    h_res=(uint*)malloc(obj_a_tri_num_*sizeof(uint)*threads_per_block);

    std::cout<<"generate tris data finish..."<<std::endl;
}

__global__ void cuTriContactDetect_v0(tri3f*,tri3f*,
                                   vec3f*,vec3f*,
                                   uint,uint,
                                   uint* res);
__global__ void cuTriContactDetect_v1(tri3f* ,tri3f*,
                                      vec3f*,vec3f*,
                                      uint, uint,
                                      uint* res);
__global__ void cuTriContactDetect_v2(vec3f*,vec3f*,uint,uint,uint*);
__global__ void cuTriContactDetect_self_v0(tri3f*,vec3f*,uint,uint*);
__global__ void cuTriContactDetect_self_v1(vec3f*,uint,uint*);
__global__ void cuTriContactDetect_bvh_v0(Triangle*,BVHNode*,uint,uint*);//堆栈
__device__ void cuIterativeCD(const Triangle *tri, const BVHNode *node, uint gid, uint &contact_num, uint *res);
__device__ bool cuTriAdjacent(const tri3f& P1,const tri3f& P2);
__device__ bool cuTriAdjacent(vec3f* v1,vec3f* v2);
__device__ bool cuTriContact(const vec3f& P1,const vec3f& P2,const vec3f& P3,const vec3f& Q1,const vec3f& Q2,const vec3f& Q3);
__device__ bool cuSurfaceContact(vec3f ax,vec3f p1, vec3f p2, vec3f p3);
__device__ bool cuRotationContact(const vec3f& ax,
                                  const vec3f& p1,const vec3f& p2,const vec3f& p3,
                                  const vec3f& q1,const vec3f& q2,const vec3f& q3);
__device__ double cuFmax(double a, double b, double c);
__device__ double cuFmin(double a, double b, double c);

#define TWO_OBJ_CD_METHOD_V2
void CollisionDetect::TriContactDetect()
{
    if(!is_self_cd){
#ifdef TWO_OBJ_CD_METHOD_V0
      START_GPU
        TriContact_v0();
      END_GPU
#elif defined(TWO_OBJ_CD_METHOD_V1)
      START_GPU
        TriContact_v1();
      END_GPU
#elif defined(TWO_OBJ_CD_METHOD_V2)
      START_GPU
        TriContact_v2();
      END_GPU
#endif
    }
    else{
        if(cd_method==CDMethod::Native){
            START_GPU
                TriContact_self_v0();
            END_GPU
        }
        else if(cd_method==CDMethod::BVH){
            START_GPU
                TriContact_self_bvh_v0();
            END_GPU
        }
    }
}

void CollisionDetect::TriContact_v0()
{
    std::cout<<"GPU Native(version 0):"<<std::endl;
    PrepareGPUData();

    blocks_per_grid=obj_a_tri_num_;

    cuTriContactDetect_v0<<<blocks_per_grid,threads_per_block>>>
            (d_obj_a_tris_,d_obj_b_tris_,
             d_obj_a_vtxs_,d_obj_b_vtxs_,
             obj_a_tri_num_,obj_b_tri_num_,
             d_res);


    CheckCudaLastError();

    checkCudaErrors(cudaMemcpy(h_res,d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block,cudaMemcpyDeviceToHost));

    FetchResult(obj_a_tri_num_);
}

void CollisionDetect::TriContact_v1()
{
    std::cout<<"GPU Native(version 1):"<<std::endl;

    PrepareGPUData();

    blocks_per_grid=(obj_a_tri_num_+threads_per_block-1)/threads_per_block;

    cuTriContactDetect_v1<<<blocks_per_grid,threads_per_block>>>
            (d_obj_a_tris_,d_obj_b_tris_,
             d_obj_a_vtxs_,d_obj_b_vtxs_,
             obj_a_tri_num_,obj_b_tri_num_,
             d_res);

    CheckCudaLastError();

    checkCudaErrors(cudaMemcpy(h_res,d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block,cudaMemcpyDeviceToHost));

    FetchResult(obj_a_tri_num_);
}

void CollisionDetect::TriContact_v2()
{
    std::cout<<"GPU Native(version 2):"<<std::endl;

    GenerateTrisData();

    blocks_per_grid=obj_a_tri_num_;

    cuTriContactDetect_v2<<<blocks_per_grid,threads_per_block>>>
        (d_obj_a_data_,d_obj_b_data_,
         obj_a_tri_num_,obj_b_tri_num_,
         d_res);

    CheckCudaLastError();

    checkCudaErrors(cudaMemcpy(h_res,d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block,cudaMemcpyDeviceToHost));

    FetchResult(obj_a_tri_num_);
}

void CollisionDetect::TriContact_self_v0()
{
    std::cout<<"GPU Naive:"<<std::endl;
    PrepareGPUData();
    blocks_per_grid=obj_a_tri_num_;

    cuTriContactDetect_self_v0<<<blocks_per_grid,threads_per_block>>>
            (d_obj_a_tris_,d_obj_a_vtxs_,obj_a_tri_num_,d_res);

    CheckCudaLastError();

    checkCudaErrors(cudaMemcpy(h_res,d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block,cudaMemcpyDeviceToHost));

    FetchResult(obj_a_tri_num_);
}

void CollisionDetect::TriContact_self_v1()
{
    GenerateTrisData();

    blocks_per_grid=obj_a_tri_num_;

    cuTriContactDetect_self_v1<<<blocks_per_grid,threads_per_block>>>
        (d_obj_a_data_,obj_a_tri_num_,d_res);

    CheckCudaLastError();

    checkCudaErrors(cudaMemcpy(h_res,d_res,sizeof(uint)*obj_a_tri_num_*threads_per_block,cudaMemcpyDeviceToHost));

    FetchResult(obj_a_tri_num_);
}

void CollisionDetect::TriContact_self_bvh_v0()
{
    PrepareGPUResultData(tree_->GetDeviceTriNum());

    blocks_per_grid=tree_->GetDeviceTriNum();

    cuTriContactDetect_bvh_v0<<<blocks_per_grid,threads_per_block>>>
            (tree_->GetDeviceTris(),tree_->GetDeviceBVHRoot(),tree_->GetDeviceTriNum(),d_res);

    CheckCudaLastError();

    checkCudaErrors(cudaMemcpy(h_res,d_res,sizeof(uint)*tree_->GetDeviceTriNum()*threads_per_block,cudaMemcpyDeviceToHost));

    FetchResult(tree_->GetDeviceTriNum());
}
void CollisionDetect::FetchResult(uint32_t tri_num)
{
    uint32_t cnt=0;
    for(int i=0;i<tri_num;i++){
        uint row=i*threads_per_block;
        if (h_res[row]>0){
            //std::cout<<"row is "<<i<<" and num is: "<<h_res[row]<<std::endl;
            cnt+=h_res[row];
            for(int j=1;j<=h_res[row];j++){
#ifdef PRINT_CDTRI_INDEX
                printf("triangle contact found at (%d, %d) = (%d, %d, %d) (%d, %d, %d)\n",
                       i,h_res[row+j],
                       obj_a_tris_[i].id0(),obj_a_tris_[i].id1(),obj_a_tris_[i].id2(),
                       obj_b_tris_[h_res[row+j]].id0(),obj_b_tris_[h_res[row+j]].id1(),obj_b_tris_[h_res[row+j]].id2());
#endif
                a_set_.insert(i);
                b_set_.insert(h_res[row+j]);

            }
        }
    }
    std::cout<<"\tGPU collision detected number is: "<<cnt<<std::endl;
}

CollisionDetect::~CollisionDetect()
{
    if(obj_a_data_) free(obj_a_data_);
    if(obj_b_data_) free(obj_b_data_);
    if(d_obj_a_tris_) cudaFree(d_obj_a_tris_);
    if(d_obj_b_tris_) cudaFree(d_obj_b_tris_);
    if(d_obj_a_vtxs_) cudaFree(d_obj_a_vtxs_);
    if(d_obj_b_vtxs_) cudaFree(d_obj_b_vtxs_);
    if(d_obj_a_data_) cudaFree(d_obj_a_data_);
    if(d_obj_b_data_) cudaFree(d_obj_b_data_);
    if(h_res) free(h_res);
    if(d_res) cudaFree(d_res);
    if(tree_) delete tree_;
}

struct Lock{
    int mutex;
    __device__ Lock(){};
    __device__ void init(){mutex=0;}
    __device__ void lock(){
        while(atomicCAS(&mutex,0,1)!=0);
    }
    __device__ void unlock(){
        atomicExch(&mutex,0);
    }
};

__global__ void cuTriContactDetect_v0(tri3f* a_tris,tri3f* b_tris,
                                   vec3f* a_vtxs,vec3f* b_vtxs,
                                   uint a_tris_num,uint b_tris_num,
                                   uint* res)
{
    __shared__ uint contacted[threads_per_block];
    __shared__ uint cur_contact_num;
    __shared__ Lock lock;

    cur_contact_num=0;

    int gid=threadIdx.x+blockIdx.x*blockDim.x;
    int bid=blockIdx.x;
    int tid=threadIdx.x;
    if (tid==0) lock.init();
    contacted[tid]=0;
    int pass=(b_tris_num+threads_per_block-1)/threads_per_block;
    uint a_tri_x=a_tris[bid].id(0);
    uint a_tri_y=a_tris[bid].id(1);
    uint a_tri_z=a_tris[bid].id(2);

    vec3f a_vtx_x=a_vtxs[a_tri_x];
    vec3f a_vtx_y=a_vtxs[a_tri_y];
    vec3f a_vtx_z=a_vtxs[a_tri_z];
    __syncthreads();
    for(int i=0;i<pass;i++){
        int idx=tid+i*threads_per_block;
        if(idx<b_tris_num){
            uint p_x=b_tris[idx].id(0);
            uint p_y=b_tris[idx].id(1);
            uint p_z=b_tris[idx].id(2);
            vec3f b_vtx_x=b_vtxs[p_x];
            vec3f b_vtx_y=b_vtxs[p_y];
            vec3f b_vtx_z=b_vtxs[p_z];
            if(cuTriContact(a_vtx_x,a_vtx_y,a_vtx_z,b_vtx_x,b_vtx_y,b_vtx_z)){
                lock.lock();
                if (cur_contact_num<threads_per_block){
                    contacted[cur_contact_num]=idx;
                    cur_contact_num++;
                }
                lock.unlock();
            }
        }
        __syncthreads();
    }
        if (tid==0){
            res[bid*threads_per_block]=cur_contact_num;
        }
        if(tid<cur_contact_num && tid<threads_per_block-1)
            res[gid+1]=contacted[tid];

}

//一个block内一个线程负责一个物体a的三角形
__global__ void cuTriContactDetect_v1(tri3f* a_tris,tri3f* b_tris,
                                      vec3f* a_vtxs,vec3f* b_vtxs,
                                      uint a_tris_num, uint b_tris_num,
                                      uint* res)
{

    __shared__ uint contact_num[threads_per_block];

    __shared__ uint contacted[threads_per_block][8];

    uint32_t gid=threadIdx.x+blockIdx.x*blockDim.x;
    uint32_t bid=blockIdx.x;
    uint32_t tid=threadIdx.x;

    uint32_t pass=(b_tris_num+threads_per_block-1)/threads_per_block;
    //read a tris from global memory
    tri3f a_tri_id=a_tris[gid];
    vec3f a_vtx_0=a_vtxs[a_tri_id.id0()];
    vec3f a_vtx_1=a_vtxs[a_tri_id.id1()];
    vec3f a_vtx_2=a_vtxs[a_tri_id.id2()];

    contact_num[tid]=0;
    __syncthreads();
    for(uint32_t i=0;i<pass;i++){
        uint pass_t=i*threads_per_block;
        uint idx=tid+pass_t;
        __shared__ double cache_b_tris[threads_per_block][9];
        tri3f b_tri_id=b_tris[idx];
        vec3f b_vtx_0=b_vtxs[b_tri_id.id0()];
        vec3f b_vtx_1=b_vtxs[b_tri_id.id1()];
        vec3f b_vtx_2=b_vtxs[b_tri_id.id2()];
        cache_b_tris[tid][0]=b_vtx_0.x;cache_b_tris[tid][1]=b_vtx_0.y;cache_b_tris[tid][2]=b_vtx_0.z;
        cache_b_tris[tid][3]=b_vtx_1.x;cache_b_tris[tid][4]=b_vtx_1.y;cache_b_tris[tid][5]=b_vtx_1.z;
        cache_b_tris[tid][6]=b_vtx_2.x;cache_b_tris[tid][7]=b_vtx_2.y;cache_b_tris[tid][8]=b_vtx_2.z;
        for(uint32_t j=0;j<threads_per_block;j++){
            uint32_t idx_=(tid+j)%threads_per_block;
            vec3f b_vtx_0_{cache_b_tris[idx_][0],cache_b_tris[idx_][1],cache_b_tris[idx_][2]};
            vec3f b_vtx_1_{cache_b_tris[idx_][3],cache_b_tris[idx_][4],cache_b_tris[idx_][5]};
            vec3f b_vtx_2_{cache_b_tris[idx_][6],cache_b_tris[idx_][7],cache_b_tris[idx_][8]};
            if(cuTriContact(a_vtx_0,a_vtx_1,a_vtx_2,b_vtx_0_,b_vtx_1_,b_vtx_2_)){
                contacted[tid][contact_num[tid]++]=pass_t+idx_;
            }
            //no need sync?
        }
        __syncthreads();
    }
    res[gid*threads_per_block]=contact_num[tid];
    for(uint32_t i=0;i<contact_num[tid];i++){
        res[gid*threads_per_block+i+1]=contacted[tid][i];
    }
}

__global__ void cuTriContactDetect_v2(vec3f* a_data,vec3f* b_data,uint a_num,uint b_num,uint* res)
{
    __shared__ uint contacted[threads_per_block];
    __shared__ uint contact_num;
    __shared__ Lock lock;
    uint32_t gid=threadIdx.x+blockIdx.x*blockDim.x;
    uint32_t bid=blockIdx.x;
    uint32_t tid=threadIdx.x;
    if(tid==0){
        contact_num=0;
        lock.init();
    }

    uint32_t pass=(b_num+threads_per_block-1)/threads_per_block;
    vec3f a_vtx_0=a_data[bid*3+0];
    vec3f a_vtx_1=a_data[bid*3+1];
    vec3f a_vtx_2=a_data[bid*3+2];
    __syncthreads();

    for(uint32_t i=0;i<pass;i++){
        uint32_t idx=tid+i*threads_per_block;
        if(idx<b_num){
            vec3f b_vtx_0=b_data[idx*3+0];
            vec3f b_vtx_1=b_data[idx*3+1];
            vec3f b_vtx_2=b_data[idx*3+2];
            if(cuTriContact(a_vtx_0,a_vtx_1,a_vtx_2,b_vtx_0,b_vtx_1,b_vtx_2)){
                lock.lock();
                if(contact_num<=MAX_CONTACT_NUM){
                    contacted[contact_num]=idx;
//                    res[bid*threads_per_block+1+contact_num]=idx;
                    contact_num++;
                }
                lock.unlock();
            }
        }
        __syncthreads();
    }
    if(tid==0)
        res[bid*threads_per_block]=contact_num;
    if(tid<contact_num && tid<MAX_CONTACT_NUM)
        res[gid+1]=contacted[tid];
}

__global__ void cuTriContactDetect_self_v0(tri3f* a_tris,vec3f* a_vtxs,uint a_tris_num,uint* res)
{

    __shared__ uint contact_num;
    __shared__ uint contacted[MAX_CONTACT_NUM];
    __shared__ Lock lock;
    uint32_t gid=threadIdx.x+blockIdx.x*blockDim.x;
    uint32_t bid=blockIdx.x;
    uint32_t tid=threadIdx.x;
//    printf("kernel\n");
    tri3f a_tri_id=a_tris[bid];
    vec3f a_vtx_0=a_vtxs[a_tri_id.id0()];
    vec3f a_vtx_1=a_vtxs[a_tri_id.id1()];
    vec3f a_vtx_2=a_vtxs[a_tri_id.id2()];
    if(tid==0){
        contact_num=0;
        lock.init();//不能多次初始化
    }
    uint32_t pass=(a_tris_num+threads_per_block-1)/threads_per_block;
    __syncthreads();

    for(uint32_t i=bid/threads_per_block;i<pass;i++){
        uint32_t idx=tid+i*threads_per_block;
        if(idx>bid && idx<a_tris_num){
            tri3f a_tri_id_=a_tris[idx];
            if(!cuTriAdjacent(a_tri_id,a_tri_id_)){
                vec3f a_vtx_0_=a_vtxs[a_tri_id_.id0()];
                vec3f a_vtx_1_=a_vtxs[a_tri_id_.id1()];
                vec3f a_vtx_2_=a_vtxs[a_tri_id_.id2()];
                if(cuTriContact(a_vtx_0,a_vtx_1,a_vtx_2,a_vtx_0_,a_vtx_1_,a_vtx_2_)){
                    lock.lock();
                    if(contact_num<MAX_CONTACT_NUM){
                        contacted[contact_num]=idx;
                        contact_num++;
                    }
                    lock.unlock();
                }
            }
        }
        __syncthreads();
    }
    if(tid==0)
        res[bid*threads_per_block]=contact_num;
    if(tid<contact_num && tid<MAX_CONTACT_NUM)
        res[gid+1]=contacted[tid];
}

__global__ void cuTriContactDetect_self_v1(vec3f* a_data,uint a_num,uint* res)
{
    __shared__ uint contact_num;
    __shared__ uint contacted[MAX_CONTACT_NUM];
    __shared__ Lock lock;
    uint32_t gid=threadIdx.x+blockIdx.x*blockDim.x;
    uint32_t bid=blockIdx.x;
    uint32_t tid=threadIdx.x;
    vec3f a_vtxs[3];
    a_vtxs[0]=a_data[bid*3+0];
    a_vtxs[1]=a_data[bid*3+1];
    a_vtxs[2]=a_data[bid*3+2];
    if(tid==0){
        contact_num=0;
        lock.init();//不能多次初始化
    }
    uint32_t pass=(a_num+threads_per_block-1)/threads_per_block;
    __syncthreads();

    for(uint32_t i=bid/threads_per_block;i<pass;i++){
        uint32_t idx=tid+i*threads_per_block;
        if(idx>bid && idx<a_num){
            vec3f a_vtxs_[3];
            a_vtxs_[0]=a_data[idx*3+0];
            a_vtxs_[1]=a_data[idx*3+1];
            a_vtxs_[2]=a_data[idx*3+2];
            if(!cuTriAdjacent(a_vtxs,a_vtxs_)){
                if(cuTriContact(a_vtxs[0],a_vtxs[1],a_vtxs[2],a_vtxs_[0],a_vtxs_[1],a_vtxs_[2])){
                    lock.lock();
                    if(contact_num<MAX_CONTACT_NUM){
                        contacted[contact_num]=idx;
                        contact_num++;
                    }
                    lock.unlock();
                }
            }
        }
        __syncthreads();
    }
    if(tid==0)
        res[bid*threads_per_block]=contact_num;
    if(tid<contact_num && tid<MAX_CONTACT_NUM)
        res[gid+1]=contacted[tid];
}

__global__ void cuTriContactDetect_bvh_v0(Triangle * tris, BVHNode * root, uint tris_num, uint* res)
{
    uint32_t gid=threadIdx.x+blockIdx.x*blockDim.x;
    uint contact_num=0;
    if(gid<tris_num){
        cuIterativeCD(&tris[gid],root,gid,contact_num,res);
        res[gid*threads_per_block]=contact_num;
    }
}
__device__ void cuIterativeCD(const Triangle *tri, const BVHNode *node, uint gid, uint &contact_num, uint *res)
{

    BVHNode* stack[64];
    uint32_t stack_pos=0;
    stack[stack_pos++]=nullptr;
    Bound3 bound=tri->CuGetBound3();
    const BVHNode* _node=node;
    do{
//        _node=nullptr;//ok
        if(_node->IsLeafNode() ){
            if(_node->object_->CuGetID()>tri->CuGetID()){
                auto v_id0=tri->CuGetTriID();
                auto v_id1=_node->object_->CuGetTriID();
                bool is_adj=false;
                for(int i=0;i<3;i++)
                    for(int j=0;j<3;j++)
                        if(v_id0.id(i)==v_id1.id(j)){
                            is_adj=true;
                            break;
                        }
               if(!is_adj && cuTriContact(tri->CuV0(),tri->CuV1(),tri->CuV2(),_node->object_->CuV0(),_node->object_->CuV1(),_node->object_->CuV2())){
                   res[gid*threads_per_block+contact_num+1]=_node->object_->CuGetID();
                   contact_num++;
               }
            }
            _node=stack[--stack_pos];
        }
        else{
            auto left=_node->left_;
            auto right=_node->right_;
            bool traverseL=Insect(bound,left->GetBound3());
            bool traverseR=Insect(bound,right->GetBound3());
            _node=(traverseL)?left:right;
            if(traverseL && traverseR)
                stack[stack_pos++]=right;
        }
    }while(_node!=nullptr);
}


__device__ bool cuTriAdjacent(const tri3f& P1,const tri3f& P2)
{
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            if(P1.id(i)==P2.id(j))
                return true;
        }
    }
    return false;
}
__device__ bool cuTriAdjacent(vec3f* v1,vec3f* v2)
{
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            if(v1[i]==v2[j])
                return true;
        }
    }
    return false;
}
__device__ bool cuTriContact(const vec3f& P1,const vec3f& P2,const vec3f& P3,
                             const vec3f& Q1,const vec3f& Q2,const vec3f& Q3)
{
    vec3f p1;
    vec3f p2=P2-P1;
    vec3f p3=P3-P1;
    vec3f q1=Q1-P1;
    vec3f q2=Q2-P1;
    vec3f q3=Q3-P1;

    vec3f e1=p2-p1;
    vec3f e2=p3-p2;
    vec3f e3=p1-p3;

    vec3f f1=q2-q1;
    vec3f f2=q3-q2;
    vec3f f3=q1-q3;

    vec3f n1=e1.cross(e2);
    vec3f m1=f1.cross(f2);

    vec3f ef11=e1.cross(f1);
    vec3f ef12=e1.cross(f2);
    vec3f ef13=e1.cross(f3);
    vec3f ef21=e2.cross(f1);
    vec3f ef22=e2.cross(f2);
    vec3f ef23=e2.cross(f3);
    vec3f ef31=e3.cross(f1);
    vec3f ef32=e3.cross(f2);
    vec3f ef33=e3.cross(f3);

    if(!cuSurfaceContact(n1,q1,q2,q3)) return false;
    if(!cuSurfaceContact(m1,-q1,p2-q1,p3-q1)) return false;

    if(!cuRotationContact(ef11,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef12,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef13,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef21,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef22,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef23,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef31,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef32,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(ef33,p1,p2,p3,q1,q2,q3)) return false;

    vec3f g1=e1.cross(n1);
    vec3f g2=e2.cross(n1);
    vec3f g3=e3.cross(n1);

    if(!cuRotationContact(g1,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(g2,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(g3,p1,p2,p3,q1,q2,q3)) return false;

    vec3f h1=f1.cross(m1);
    vec3f h2=f2.cross(m1);
    vec3f h3=f3.cross(m1);

    if(!cuRotationContact(h1,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(h2,p1,p2,p3,q1,q2,q3)) return false;
    if(!cuRotationContact(h3,p1,p2,p3,q1,q2,q3)) return false;

    return true;
}

__device__ bool cuSurfaceContact(vec3f ax,
                                 vec3f p1, vec3f p2, vec3f p3)
{
    double P1 = ax.dot(p1);
    double P2 = ax.dot(p2);
    double P3 = ax.dot(p3);

    double mx1 = cuFmax(P1, P2, P3);
    double mn1 = cuFmin(P1, P2, P3);

    if (mn1 > 0.0) return false;
    if (mx1 < 0.0) return false;

    return true;
}

__device__ bool cuRotationContact(const vec3f& ax,
                                  const vec3f& p1, const vec3f& p2, const vec3f& p3,
                                  const vec3f& q1, const vec3f& q2, const vec3f& q3)
{
    double P1 = ax.dot(p1);
    double P2 = ax.dot(p2);
    double P3 = ax.dot(p3);
    double Q1 = ax.dot(q1);
    double Q2 = ax.dot(q2);
    double Q3 = ax.dot(q3);

    double mx1 = cuFmax(P1, P2, P3);
    double mn1 = cuFmin(P1, P2, P3);
    double mx2 = cuFmax(Q1, Q2, Q3);
    double mn2 = cuFmin(Q1, Q2, Q3);

    if (mn1 > mx2) return false;
    if (mn2 > mx1) return false;
    return true;
}
__device__ double cuFmax(double a, double b, double c)
{
    double t = a;
    if (b > t) t = b;
    if (c > t) t = c;
    return t;
}
__device__ double cuFmin(double a, double b, double c)
{
    double t = a;
    if (b < t) t = b;
    if (c < t) t = c;
    return t;
}



