#include "runtime.h"
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <cstring>
#include <string>
#include <unordered_map>

struct Model {
    int size,channels,blocks;
    at::Device device;
    std::unordered_map<std::string,at::Tensor> weights;
    Model(int n,int c,int b,const char* name) : size(n),channels(c),blocks(b),device(name) {}
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
    std::pair<at::Tensor,at::Tensor> forward(const at::Tensor& input) const {
        auto x=at::relu(norm(conv(input,"trunk.0",1),"trunk.1"));
        for (int i=0; i<blocks; ++i) {
            std::string prefix="trunk."+std::to_string(i+3)+".layers.";
            auto y=at::relu(norm(conv(x,prefix+"0",1),prefix+"1"));
            x=at::relu(x+norm(conv(y,prefix+"3",1),prefix+"4"));
        }
        auto policy=conv(x,"policy",0,true).flatten(1);
        auto value=at::relu(conv(x,"value.0",0,true)).flatten(1);
        value=at::relu(at::linear(value,get("value.3.weight"),get("value.3.bias")));
        value=at::tanh(at::linear(value,get("value.5.weight"),get("value.5.bias"))).squeeze(1);
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
EXPORT void* model_create(int size,int channels,int blocks,const char* device) {
    try {
        if (size<2 || size>25 || channels<4 || channels>256 || blocks<0 || blocks>32) throw std::invalid_argument("Invalid model dimensions.");
        return new Model(size,channels,blocks,device);
    } catch (const std::exception& e) { set_error(e.what()); return nullptr; }
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
