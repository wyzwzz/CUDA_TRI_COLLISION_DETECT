//
// Created by wyz on 20-11-6.
//

#ifndef OBJCDCHECKER_BVH_H
#define OBJCDCHECKER_BVH_H

#include<vector>
#include<limits>
#include"cmesh.h"

class Bound3{
public:
    vec3f p_min_,p_max_;
public:
    __host__ __device__ Bound3(){
        double min_double_=std::numeric_limits<double>::lowest();
        double max_double_=std::numeric_limits<double>::max();
        p_max_=vec3f(min_double_);
        p_min_=vec3f(max_double_);
    }
    __host__ __device__ Bound3(const Bound3& b):p_min_(b.p_min_),p_max_(b.p_max_){}
    __host__ __device__ Bound3(const vec3f& p1,const vec3f& p2){
        p_min_=vec3f(fmin(p1.x,p2.x),fmin(p1.y,p2.y),fmin(p1.z,p2.z));
        p_max_=vec3f(fmax(p1.x,p2.x),fmax(p1.y,p2.y),fmax(p1.z,p2.z));
    }
    __host__ __device__ Bound3(const vec3f& p1,const vec3f& p2,const vec3f& p3){
        p_min_=vec3f(fmin(p1.x,fmin(p2.x,p3.x)),fmin(p1.y,fmin(p2.y,p3.y)),fmin(p1.z,fmin(p2.z,p3.z)));
        p_max_=vec3f(fmax(p1.x,fmax(p2.x,p3.x)),fmax(p1.y,fmax(p2.y,p3.y)),fmax(p1.z,fmax(p2.z,p3.z)));
    }
    __host__ __device__ vec3f Diagonal() const{return p_max_-p_min_;}
    __host__ __device__ int MaxExtent() const{
        vec3f d=Diagonal();
        if(d.x>d.y && d.x>d.z)
            return 0;
        else if(d.y>d.z)
            return 1;
        else return 2;
    }
    __host__ __device__ double SurfaceArea() const{
        vec3f d=Diagonal();
        return 2*(d.x*d.y+d.x*d.z+d.y*d.z);
    };
    __host__ __device__ vec3f CentroID() const{return (p_min_+p_max_)*0.5;}
    __host__ __device__ Bound3 Intersect(const Bound3& b){
        return Bound3(vec3f(fmax(p_min_.x,b.p_min_.x),
                               fmax(p_min_.y,b.p_min_.y),
                               fmax(p_min_.z,b.p_min_.z)),
                      vec3f(fmin(p_max_.x,b.p_max_.x),
                               fmin(p_max_.y,b.p_max_.y),
                               fmin(p_max_.z,b.p_max_.z)));
    }
    __host__ __device__ vec3f Offset(const vec3f& p) const{
        vec3f o=p-p_min_;
        if(p_max_.x>p_min_.x)
            o.x/=p_max_.x-p_min_.x;
        if(p_max_.y>p_min_.y)
            o.y/=p_max_.y-p_min_.y;
        if(p_max_.z>p_min_.z)
            o.z/=p_max_.z-p_min_.z;
        return o;
    }
    __host__ __device__ const vec3f& operator[](int i) const{
        return (i==0)?p_min_:p_max_;
    }
    __host__ __device__ Bound3 Union(const Bound3& b){
        return Bound3(vec3f(fmin(p_min_.x,b.p_min_.x),
                               fmin(p_min_.y,b.p_min_.y),
                               fmin(p_min_.z,b.p_min_.z)),
                      vec3f(fmax(p_max_.x,b.p_max_.x),
                               fmax(p_max_.y,b.p_max_.y),
                               fmax(p_max_.z,b.p_max_.z)));
    }
    __host__ __device__ Bound3 Union(const vec3f& p){
        return Bound3(vec3f(fmin(p_min_.x,p.x),
                               fmin(p_min_.y,p.y),
                               fmin(p_min_.z,p.z)),
                      vec3f(fmax(p_max_.x,p.x),
                               fmax(p_max_.y,p.y),
                               fmax(p_max_.z,p.z)));
    }
};
__host__ __device__ inline Bound3 Union(const Bound3& b1,const Bound3& b2)
{
    return Bound3(vec3f(fmin(b1.p_min_.x,b2.p_min_.x),
                        fmin(b1.p_min_.y,b2.p_min_.y),
                        fmin(b1.p_min_.z,b2.p_min_.z)),
                  vec3f(fmax(b1.p_max_.x,b2.p_max_.x),
                        fmax(b1.p_max_.y,b2.p_max_.y),
                        fmax(b1.p_max_.z,b2.p_max_.z)));
}
__host__ __device__ inline Bound3 Union(const Bound3& b,const vec3f& p)
{
    return Bound3(vec3f(fmin(b.p_min_.x,p.x),
                        fmin(b.p_min_.y,p.y),
                        fmin(b.p_min_.z,p.z)),
                  vec3f(fmax(b.p_max_.x,p.x),
                        fmax(b.p_max_.y,p.y),
                        fmax(b.p_max_.z,p.z)));
}
__host__ __device__ inline bool Insect(const Bound3& b1,const Bound3& b2)
{
    if(    fmax(b1.p_min_.x,b2.p_min_.x)>fmin(b1.p_max_.x,b2.p_max_.x) ||
           fmax(b1.p_min_.y,b2.p_min_.y)>fmin(b1.p_max_.y,b2.p_max_.y) ||
           fmax(b1.p_min_.z,b2.p_min_.z)>fmin(b1.p_max_.z,b2.p_max_.z)  )
        return false;
    return true;
}
class Object{
public:
    __host__ __device__ Object()=default;
    __host__ __device__ virtual ~Object()=default;
    __host__ __device__ virtual const Bound3& GetBound3() const=0;
    __host__ __device__ virtual const tri3f& GetTriID() const=0;
    __host__ __device__ virtual const vec3f& V0() const=0;
    __host__ __device__ virtual const vec3f& V1() const=0;
    __host__ __device__ virtual const vec3f& V2() const=0;
    __host__ __device__ virtual uint32_t GetID() const=0;

};
class Triangle: public Object{
    vec3f v0_,v1_,v2_;

    Bound3 bound3;
public:
    tri3f v_id_;//用于排除三角形自交
    uint32_t id;
    __host__ Triangle(){}
    __host__ __device__ Triangle(const vec3f& v0,const vec3f& v1,const vec3f v2)
    :v0_(v0),v1_(v1),v2_(v2),bound3(v0,v1,v2)
    {}
    __host__ __device__ const vec3f& V0() const{return v0_;}
    __host__ __device__ const vec3f& V1() const{return v1_;}
    __host__ __device__ const vec3f& V2() const{return v2_;}
    __host__ __device__ const Bound3& GetBound3()const{
        return  bound3;
    }
    __device__ const vec3f& CuV0() const{return v0_;}
    __device__ const vec3f& CuV1() const{return v1_;}
    __device__ const vec3f& CuV2() const{return v2_;}
    __device__ const Bound3& CuGetBound3()const{
        return  bound3;
    }
    __host__ __device__ const tri3f& GetTriID() const{
        return v_id_;
    }
    __device__ const tri3f& CuGetTriID() const{
        return v_id_;
    }
    __host__ __device__ uint32_t GetID() const override{
        return id;
    }
    __device__ uint32_t CuGetID() const{
        return id;
    }
    void SetVertex(const vec3f& v0,const vec3f& v1,const vec3f& v2){
        v0_=v0;
        v1_=v1;
        v2_=v2;
        bound3=Bound3(v0,v1,v2);
    }
};
class BVHNode{
public:
    Bound3 bound_;
    BVHNode* left_;
    BVHNode* right_;
    Triangle* object_;
//    int32_t split_axis_=0;
//    int32_t first_prim_offset=0;
//    int32_t n_primitives=0;
    __host__ __device__ BVHNode():bound_(Bound3()),left_(nullptr),right_(nullptr),object_(nullptr){}
    __host__ __device__ bool IsLeafNode() const{
        return (left_== nullptr && right_== nullptr);
    }
    __host__ __device__ const Bound3& GetBound3() const{
        return bound_;
    }
};
class CuBVH_Impl;
class BVH final{
public:
    enum class SplitMethod{NAIVE};
    enum class DeviceType{CPU,GPU};
    __host__ BVH(mesh* m,set<int>& cd_res,uint32_t max_prims_in_node,DeviceType device_type=DeviceType::CPU,SplitMethod split_method=SplitMethod::NAIVE);
    ~BVH();
    __host__ BVHNode* RecursiveBuild(){RecursiveBuild(tris_);}
    __host__ void TriangleSelfCD();
    __host__ void CuBVHBuild();

    Triangle* GetDeviceTris() const;
    Triangle* GetHostTris() const;
    BVHNode* GetDeviceBVHRoot() const;
    uint32_t GetDeviceTriNum() const;
public:
    __host__ BVHNode* RecursiveBuild(std::vector<Triangle*>&);
    bool RecursiveCD(const Triangle* tri,const BVHNode* node);
    void RecursiveFree(BVHNode* node);

    BVHNode* root_;
    const uint32_t max_prims_in_node_;
    const SplitMethod split_method_;
    const DeviceType device_type_;
    mesh* mesh_;
    std::vector<Triangle*> tris_;
    uint32_t tris_num_;
    set<int>& cd_res_;
    //device
    CuBVH_Impl* impl_;
};


#endif //OBJCDCHECKER_BVH_H
