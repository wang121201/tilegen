#pragma once
#include <array>
#if defined(__APPLE__) && !defined(TINY_SHA_PORTABLE)
#include <CommonCrypto/CommonDigest.h>
#endif
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
namespace tiny_sha {
using U=std::uint64_t;
inline void need(bool x,const std::string& why){if(!x)throw std::invalid_argument(why);}
template<std::size_t N> inline void need(bool x,const char (&why)[N]){if(!x)throw std::invalid_argument(why); }
// Same streaming SHA-256 contract; platform acceleration with the original portable fallback.
#if defined(__APPLE__) && !defined(TINY_SHA_PORTABLE)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
class Sha256 {
    CC_SHA256_CTX state{};
    U length=0;
public:
    Sha256(){need(CC_SHA256_Init(&state)==1,"system SHA256 initialization");}
    void add(const void* data,std::size_t size){
        need(size<=UINT64_MAX-length,"hash byte count overflow");
        need(data!=nullptr||size==0,"null nonempty hash input");
        length+=size;
        auto p=static_cast<const unsigned char*>(data);
        while(size){
            const auto n=static_cast<CC_LONG>(std::min(size,std::size_t(std::numeric_limits<CC_LONG>::max())));
            need(CC_SHA256_Update(&state,p,n)==1,"system SHA256 update");
            p+=n;size-=n;
        }
    }
    void add(const std::string& s){add(s.data(),s.size());}
    U bytes()const{return length;}
    std::string hex()const{
        need(length<=UINT64_MAX/8,"hash bit count overflow");
        auto copy=state;
        unsigned char digest[CC_SHA256_DIGEST_LENGTH];
        need(CC_SHA256_Final(digest,&copy)==1,"system SHA256 finalization");
        constexpr char digits[]="0123456789abcdef";
        std::string out(2*CC_SHA256_DIGEST_LENGTH,'0');
        for(std::size_t i=0;i<CC_SHA256_DIGEST_LENGTH;++i){out[2*i]=digits[digest[i]>>4];out[2*i+1]=digits[digest[i]&15];}
        return out;
    }
};
#pragma clang diagnostic pop
#else
class Sha256 {
    std::array<std::uint32_t,8> h{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    std::array<unsigned char,64> b{};std::size_t used=0;U length=0;
    static std::uint32_t rotr(std::uint32_t x,int n){return (x>>n)|(x<<(32-n));}
    void block(){
        static constexpr std::uint32_t k[64]={
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        std::uint32_t w[64];for(int i=0;i<16;++i)w[i]=(std::uint32_t(b[4*i])<<24)|(std::uint32_t(b[4*i+1])<<16)|(std::uint32_t(b[4*i+2])<<8)|b[4*i+3];
        for(int i=16;i<64;++i){auto x=w[i-15],y=w[i-2];w[i]=w[i-16]+(rotr(x,7)^rotr(x,18)^(x>>3))+w[i-7]+(rotr(y,17)^rotr(y,19)^(y>>10));}
        auto a=h[0],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],v=h[7];auto z=h[1];
        for(int i=0;i<64;++i){auto t1=v+(rotr(e,6)^rotr(e,11)^rotr(e,25))+((e&f)^(~e&g))+k[i]+w[i];auto t2=(rotr(a,2)^rotr(a,13)^rotr(a,22))+((a&z)^(a&c)^(z&c));v=g;g=f;f=e;e=d+t1;d=c;c=z;z=a;a=t1+t2;}
        h[0]+=a;h[1]+=z;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=v;
    }
public:
    // Fixed byte-span input. Copy only bounded blocks into the original compression
    // buffer; block(), final padding, message length and digest encoding are unchanged.
    void add(const void* data,std::size_t size){
        need(size<=UINT64_MAX-length,"hash byte count overflow");
        need(data!=nullptr||size==0,"null nonempty hash input");
        length+=size;
        if(!size)return;
        auto p=static_cast<const unsigned char*>(data);
        if(used){
            need(used<b.size(),"partial hash block invariant");
            const auto available=b.size()-used;
            if(size<available){std::memcpy(b.data()+used,p,size);used+=size;return;}
            std::memcpy(b.data()+used,p,available);p+=available;size-=available;block();used=0;
        }
        while(size>=b.size()){std::memcpy(b.data(),p,b.size());block();p+=b.size();size-=b.size();}
        if(size){std::memcpy(b.data(),p,size);used=size;}
    }
    void add(const std::string& s){add(s.data(),s.size());}
    U bytes()const{return length;}
    std::string hex()const{
        auto q=*this;need(length<=UINT64_MAX/8,"hash bit count overflow");U bits=length*8;
        need(q.used<q.b.size(),"final hash block invariant");
        q.b[q.used++]=0x80;
        if(q.used>56){while(q.used<q.b.size())q.b[q.used++]=0;q.block();q.used=0;}
        while(q.used<56)q.b[q.used++]=0;
        for(int i=7;i>=0;--i)q.b[q.used++]=static_cast<unsigned char>(bits>>(i*8));
        q.block();
        std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto word:q.h)out<<std::setw(8)<<word;return out.str();
    }
};
#endif
inline std::string sha256(const std::string& bytes){Sha256 s;s.add(bytes);return s.hex();}

}
