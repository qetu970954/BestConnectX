#include "runtime.h"
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <cstring>
#include <string>
#include <unordered_map>

struct Model {
    int size,channels,blocks,architecture; // 0: residual, 1: pooled value, 2: pooled value + attention.
    at::Device device;
    at::Tensor relative_index;
    std::unordered_map<std::string,at::Tensor> weights;
    Model(int n,int c,int b,int a,const char* name) : size(n),channels(c),blocks(b),architecture(a),device(name) {
        if (architecture==2) {
            auto cells=at::arange(n*n,at::TensorOptions().dtype(at::kLong).device(device));
            auto rows=at::floor_divide(cells,n), columns=at::remainder(cells,n);
            relative_index=((rows.unsqueeze(1)-rows.unsqueeze(0)+n-1)*(2*n-1)
                            +columns.unsqueeze(1)-columns.unsqueeze(0)+n-1).flatten();
        }
    }
    const at::Tensor& get(const std::string& name) const {
        auto found=weights.find(name);
        if (found==weights.end()) throw std::invalid_argument("Missing model tensor: "+name);
        return found->second;
    }
    at::Tensor conv(const at::Tensor& x,const std::string& name,int padding,bool bias=false) const {
        return at::conv2d(x,get(name+".weight"),bias ? std::optional<at::Tensor>(get(name+".bias")) : std::nullopt,
                          at::IntArrayRef{1,1},at::IntArrayRef{padding,padding},at::IntArrayRef{1,1},1);
    }
    at::Tensor norm(const at::Tensor& x,const std::string& name) const {
        return at::batch_norm(x,get(name+".weight"),get(name+".bias"),get(name+".running_mean"),get(name+".running_var"),false,.1,1e-5,true);
    }
    at::Tensor linear(const at::Tensor& x,const std::string& name) const {
        return at::linear(x,get(name+".weight"),get(name+".bias"));
    }
    at::Tensor layernorm(const at::Tensor& x,const std::string& name) const {
        return at::layer_norm(x,{channels},get(name+".weight"),get(name+".bias"),1e-5);
    }
    at::Tensor attend(const at::Tensor& x) const {
        auto batch=x.size(0);
        int area=size*size;
        auto tokens=x.flatten(2).transpose(1,2);
        auto qkv=linear(layernorm(tokens,"attention.norm1"),"attention.qkv")
                     .reshape({batch,area,3,4,channels/4}).permute({2,0,3,1,4});
        auto bias=get("attention.relative_bias").index_select(1,relative_index).reshape({4,area,area});
        auto attended=at::scaled_dot_product_attention(qkv[0],qkv[1],qkv[2],bias)
                          .transpose(1,2).reshape({batch,area,channels});
        tokens=tokens+linear(attended,"attention.projection");
        tokens=tokens+linear(at::gelu(linear(layernorm(tokens,"attention.norm2"),"attention.feedforward.0")),
                             "attention.feedforward.2");
        return tokens.transpose(1,2).reshape({batch,channels,size,size});
    }
    std::pair<at::Tensor,at::Tensor> forward(const at::Tensor& input) const {
        auto x=at::relu(norm(conv(input,"trunk.0",1),"trunk.1"));
        for (int i=0; i<blocks; ++i) {
            std::string prefix="trunk."+std::to_string(i+3)+".layers.";
            auto y=at::relu(norm(conv(x,prefix+"0",1),prefix+"1"));
            x=at::relu(x+norm(conv(y,prefix+"3",1),prefix+"4"));
        }
        if (architecture==2) x=attend(x);
        auto policy=conv(x,"policy",0,true).flatten(1);
        auto value=at::relu(conv(x,"value.0",0,true));
        if (architecture!=0) value=value.mean({2,3},true);
        value=at::relu(linear(value.flatten(1),"value.3"));
        value=at::tanh(linear(value,"value.5")).squeeze(1);
        return {policy,value};
    }
};
void predict(void* handle,const float* input,int batch,int size,float* policy,float* value) {
    if (!handle || batch<1 || batch>128) throw std::invalid_argument("Invalid inference batch.");
    auto& model=*static_cast<Model*>(handle);
    if (size!=model.size) throw std::invalid_argument("Model and board sizes differ.");
    c10::InferenceMode guard;
    auto x=at::from_blob(const_cast<float*>(input),{batch,8,size,size},at::TensorOptions().dtype(at::kFloat)).to(model.device);
    auto result=model.forward(x);
    auto p=result.first.to(at::kCPU).contiguous(), v=result.second.to(at::kCPU).contiguous();
    std::memcpy(policy,p.const_data_ptr<float>(),static_cast<std::size_t>(batch)*size*size*sizeof(float));
    std::memcpy(value,v.const_data_ptr<float>(),static_cast<std::size_t>(batch)*sizeof(float));
}
EXPORT void* model_create_variant(int size,int channels,int blocks,int architecture,const char* device) {
    try {
        if (size<2 || size>25 || channels<4 || channels>256 || blocks<0 || blocks>32
            || architecture<0 || architecture>2 || (architecture==2 && channels%4) || !device)
            throw std::invalid_argument("Invalid model dimensions or architecture.");
        return new Model(size,channels,blocks,architecture,device);
    } catch (const std::exception& e) { set_error(e.what()); return nullptr; }
}
EXPORT void* model_create(int size,int channels,int blocks,const char* device) {
    return model_create_variant(size,channels,blocks,0,device);
}
EXPORT void model_destroy(void* handle) { delete static_cast<Model*>(handle); }
EXPORT int model_tensor(void* handle,const char* name,const float* data,const std::int64_t* shape,int rank) {
    try {
        if (!handle || rank<1 || rank>4) throw std::invalid_argument("Invalid model tensor.");
        for (int i=0; i<rank; ++i) if (shape[i]<1) throw std::invalid_argument("Invalid tensor shape.");
        auto& model=*static_cast<Model*>(handle); c10::InferenceMode guard;
        auto tensor=at::from_blob(const_cast<float*>(data),at::IntArrayRef(shape,rank),at::TensorOptions().dtype(at::kFloat)).clone().to(model.device);
        if (!at::isfinite(tensor).all().item<bool>()) throw std::invalid_argument("Non-finite model weights.");
        model.weights[name]=std::move(tensor); return 0;
    } catch (const std::exception& e) { set_error(e.what()); return -1; }
}
EXPORT int model_predict(void* handle,const float* input,int batch,int size,float* policy,float* value) {
    try { predict(handle,input,batch,size,policy,value); return 0; }
    catch (const std::exception& e) { set_error(e.what()); return -1; }
}
