#pragma once

#include <array>

namespace {
template <class T> inline std::array<T,3> myCross(const std::array<T,3>& v1,const std::array<T,3>& v2) {
  return std::array<T,3>{v1[1]*v2[2]-v1[2]*v2[1],
                      v1[2]*v2[0]-v1[0]*v2[2],
                      v1[0]*v2[1]-v1[1]*v2[0]};
}

template <class T> inline T myCross(const std::array<T,2>& v1,const std::array<T,2>& v2) {
  return v1[0]*v2[1]-v1[1]*v2[0];
}

template<class T, std::size_t D>
std::array<T, D> myNormalize(std::array<T, D> v) {
  T sum{};
  for(const auto &it : v) {
    sum += it * it;
  }
  sum = std::sqrt(sum);
  for(auto &it : v) {
    it /= sum;
  }
  return v;
}

template <class T, std::size_t D> inline T dot(const std::array<T,D>& v1, const std::array<T,D>& v2) {
  T sum(0);
  for (int i=0;i<D;i++) sum+=v1[i]*v2[i];
  return sum;
}

template<class T, std::size_t D>
std::array<T, D> myMulti(std::array<T, D> arr, T factor) {
  for(auto &it : arr) {
    it *= factor;
  }
  return arr;
}

template<class T, std::size_t D>
std::array<T, D> mySub(std::array<T, D> a1, std::array<T, D> a2) {
  for(unsigned i = 0; i < D; ++i) {
    a1[i] -= a2[i];
  }
  return a1;
}

}


template<class T,int D>
T fourRateInterpolation(std::array<T,D> nv, std::array<T,3> direction100, std::array<T,3> direction010, T r100, T r110, T r111, T r311 ){

    T Velocity=0;
    std::array<T,3> directions[3];

    directions[0] = direction100;
    directions[1] = direction010;

    directions[0] = myNormalize(directions[0]);
    directions[1] = myNormalize(
        mySub(directions[1], myMulti(directions[0], dot(directions[0],directions[1]))));
    directions[2] = myCross(directions[0], directions[1]);

    std::array<T,3> NormalVector;
    NormalVector[0] = nv[0];
    NormalVector[1] = nv[1];
    if(D==3){
      NormalVector[2] = nv[2];
    }else{
      NormalVector[2] = 0;
    }

    NormalVector=myNormalize(NormalVector);

    // for (int i=0;i<3;++i) assert(dot(directions[i], directions[(i+1)%3])<1e-6);

    std::array<T,3> N;

    for (int i=0;i<3;i++) N[i]=std::fabs(directions[i][0]*NormalVector[0]+directions[i][1]*NormalVector[1]+directions[i][2]*NormalVector[2]);

    std::sort(N.begin(), N.end(), std::greater<T>());

    // assert(std::fabs(Norm(N)-1)<1e-4);

    auto temp = std::array<T,3>{-1,1,2};
    if (dot(N, temp) < 0) {
        Velocity=-((r100*(N[0]-N[1]-2*N[2])+r110*(N[1]-N[2])+3*r311*N[2])/N[0]);    //region A
    } else {
        Velocity=-((r111*((N[1]-N[0])*0.5+N[2])+r110*(N[1]-N[2])+1.5*r311*(N[0]-N[1]))/N[0]);//region C
    }

    return Velocity;
}

