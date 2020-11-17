//
// Created by wyz on 20-11-2.
//

#ifndef OBJCDCHECKER_CD_GPU_CUH
#define OBJCDCHECKER_CD_GPU_CUH


#include<helper_cuda.h>
#include"cmesh.h"
#include <vector>
#include<iostream>
#include"bvh.h"

#define HANDLE_ERROR checkCudaErrors
#define START_GPU {\
cudaEvent_t     start, stop;\
float   elapsedTime;\
checkCudaErrors(cudaEventCreate(&start)); \
checkCudaErrors(cudaEventCreate(&stop));\
checkCudaErrors(cudaEventRecord(start, 0));\

#define END_GPU \
checkCudaErrors(cudaEventRecord(stop, 0));\
checkCudaErrors(cudaEventSynchronize(stop));\
checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop)); \
printf("\tGPU collision detected time used: %.f ms\n", elapsedTime);\
checkCudaErrors(cudaEventDestroy(start));\
checkCudaErrors(cudaEventDestroy(stop));}

#define START_CPU {\
double start = omp_get_wtime();

#define END_CPU \
double end = omp_get_wtime();\
double duration = end - start;\
printf("\tCPU collision detected time used: %.f ms\n", duration * 1000);}

class CollisionDetect{
public:
    enum class CDMethod{BVH,Native};
private:
    mesh *obj_a_,*obj_b_;
    set<int>& a_set_;
    set<int>& b_set_;
    //顶点数目
    uint obj_a_tri_num_,obj_b_tri_num_;
    //三角形数目
    uint obj_a_vtx_num_,obj_b_vtx_num_;
    //host
    tri3f *obj_a_tris_,*obj_b_tris_;
    vec3f *obj_a_vtxs_,*obj_b_vtxs_;
    vec3f *obj_a_data_,*obj_b_data_;
    //device
    tri3f *d_obj_a_tris_,*d_obj_b_tris_;
    vec3f *d_obj_a_vtxs_,*d_obj_b_vtxs_;
    vec3f *d_obj_a_data_,*d_obj_b_data_;
    //using bvh with two version: cpu and gpu
    //bvh相关的数据变量全部在bvh树当中
    BVH* tree_;
    //result
    uint* d_res;
    uint* h_res;
    double* d_test,*h_test;

    const bool is_self_cd;
    CDMethod cd_method;
public:

    explicit CollisionDetect(mesh* obj_a,mesh* obj_b,set<int>& a_set,set<int>& b_set)
    :obj_a_(obj_a),obj_b_(obj_b),a_set_(a_set),b_set_(b_set),is_self_cd(false),cd_method(CDMethod::Native),
    obj_a_data_(nullptr),obj_b_data_(nullptr),
     d_obj_a_tris_(nullptr),d_obj_b_tris_(nullptr),d_obj_a_vtxs_(nullptr),d_obj_b_vtxs_(nullptr),
     d_obj_a_data_(nullptr),d_obj_b_data_(nullptr),tree_(nullptr)
    {
        assert(obj_a_ && obj_b_);
        if(obj_a_->getNbFaces()<obj_b_->getNbFaces()){
            std::swap(obj_a_,obj_b_);
            std::swap(a_set_,b_set_);
        }
        obj_a_tri_num_=obj_a_->getNbFaces();
        obj_a_vtx_num_=obj_a_->getNbVertices();
        obj_b_tri_num_=obj_b_->getNbFaces();
        obj_b_vtx_num_=obj_b_->getNbVertices();
        obj_a_tris_=obj_a_->_tris;
        obj_a_vtxs_=obj_a_->_vtxs;
        obj_b_tris_=obj_b_->_tris;
        obj_b_vtxs_=obj_b_->_vtxs;
    }
    explicit CollisionDetect(mesh* obj_a,set<int>& a_set,CDMethod cd_method)
    :obj_a_(obj_a),a_set_(a_set),b_set_(a_set),is_self_cd(true),cd_method(cd_method),
     obj_a_data_(nullptr),obj_b_data_(nullptr),
     d_obj_a_tris_(nullptr),d_obj_b_tris_(nullptr),d_obj_a_vtxs_(nullptr),d_obj_b_vtxs_(nullptr),
     d_obj_a_data_(nullptr),d_obj_b_data_(nullptr),tree_(nullptr)
    {
        assert(obj_a);
        if(cd_method==CDMethod::Native){
            obj_a_tri_num_=obj_a_->getNbFaces();
            obj_a_vtx_num_=obj_a_->getNbVertices();
            obj_a_tris_=obj_a_->_tris;
            obj_a_vtxs_=obj_a_->_vtxs;
        }
        else if(cd_method==CDMethod::BVH){
            tree_=new BVH(obj_a,a_set,1,BVH::DeviceType::GPU);
            obj_a_tris_=obj_b_tris_=obj_a->_tris;
        }
    }

    CollisionDetect(mesh* m,set<int>& cd_res,uint32_t max_prim_in_node):
    a_set_(cd_res),b_set_(cd_res),is_self_cd(true)
    {
        tree_=new BVH(m,cd_res,1,BVH::DeviceType::GPU);
        obj_a_tris_=obj_b_tris_=m->_tris;
    }
    ~CollisionDetect();
    void TriContactDetect();
    void TriContactDetect(const tri3f* obj_a_tris,const tri3f* obj_b_tris,const vec3f* obj_a_vtxs,const vec3f* obj_b_vtxs)=delete;
    bool TriContact(const vec3f& P1,const vec3f& P2,const vec3f& P3,const vec3f& Q1,const vec3f& Q2,const vec3f& Q3)=delete;
    bool SurfaceContact(const vec3f& ax,const vec3f& p1,const vec3f& p2,const vec3f& p3)=delete;
    bool RotationContact(const vec3f& ax,const vec3f& p1,const vec3f& p2,const vec3f& p3,const vec3f& q1,const vec3f& q2,const vec3f& q3)=delete;
private:
    void TriContact_v0();
    void TriContact_v1();
    void TriContact_v2();
    void TriContact_v3();
    void TriContact_self_v0();
    void TriContact_self_v1();
    void TriContact_self_bvh_v0();

    void PrepareGPUData();
    void GenerateTrisData();
    void PrepareGPUResultData(uint32_t tri_num);//use for bvh

    void CheckCudaLastError(){
        auto err=cudaGetLastError();
        if(err!=cudaSuccess){
            throw runtime_error("kernel error code: "+to_string(err));
        }
    }
    void FetchResult(uint32_t tri_num);
};



#endif //OBJCDCHECKER_CD_GPU_CUH
