#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"
#include "kernel/norm.h"
#include <unistd.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

namespace triton { namespace backend { namespace image_process {

#define RESPOND_AND_RETURN_IF_ERROR(REQUEST, X)                         \
  do {                                                                  \
    TRITONSERVER_Error* rarie_err__ = (X);                              \
    if (rarie_err__ != nullptr) {                                       \
      TRITONBACKEND_Response* rarie_response__ = nullptr;               \
      LOG_IF_ERROR(                                                     \
          TRITONBACKEND_ResponseNew(&rarie_response__, REQUEST),        \
          "failed to create response");                                 \
      if (rarie_response__ != nullptr) {                                \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                rarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                rarie_err__),                                           \
            "failed to send error response");                           \
      }                                                                 \
      TRITONSERVER_ErrorDelete(rarie_err__);                            \
      return rarie_err__;                                               \
    }                                                                   \
  } while (false)                                                       \


#define RESPOND_FACTORY_AND_RETURN_IF_ERROR(FACTORY, X)                      \
  do {                                                                       \
    TRITONSERVER_Error* rfarie_err__ = (X);                                  \
    if (rfarie_err__ != nullptr) {                                           \
      TRITONBACKEND_Response* rfarie_response__ = nullptr;                   \
      LOG_IF_ERROR(                                                          \
          TRITONBACKEND_ResponseNewFromFactory(&rfarie_response__, FACTORY), \
          "failed to create response");                                      \
      if (rfarie_response__ != nullptr) {                                    \
        LOG_IF_ERROR(                                                        \
            TRITONBACKEND_ResponseSend(                                      \
                rfarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL,     \
                rfarie_err__),                                               \
            "failed to send error response");                                \
      }                                                                      \
      TRITONSERVER_ErrorDelete(rfarie_err__);                                \
      return rfarie_err__;                                                    \
    }                                                                        \
  } while (false)   

extern "C" {
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
          .c_str());

  delete state;

  return nullptr;  // success
}

}  
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  const std::string& InputTensorName() const { return input_name_; }

  const std::string& OutputTensorName() const { return output_name_; }

  TRITONSERVER_DataType TensorDataType() const { return inp_datatype_; }

  TRITONSERVER_DataType TensorOutDataType() const { return out_datatype_; }

  float3 NormMean() const { return float3{float(mean_[0]), float(mean_[1]), float(mean_[2])};}

  float3 NormStd() const { return float3{float(std_[0]), float(std_[1]), float(std_[2])};}

  std::vector<int64_t> TargetShape() const {return  output_shape_;}

  std::vector<int64_t> InputShape() const {return input_shape_;}

  TRITONSERVER_Error* TensorShape(std::vector<int64_t>& shape);

  TRITONSERVER_Error* ValidateModelConfig();

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  std::string input_name_;
  std::string output_name_;

  TRITONSERVER_DataType inp_datatype_;
  TRITONSERVER_DataType out_datatype_;

  std::vector<double> std_;
  std::vector<double> mean_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> input_shape_;
};

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
}

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::TensorShape(std::vector<int64_t>& shape)
{
  shape = output_shape_;
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("model configuration:\n") + buffer.Contents()).c_str());
  }
  common::TritonJson::Value inputs, outputs, parameters;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));
  RETURN_ERROR_IF_FALSE(ModelConfig().Find("parameters", &parameters), TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model config find parameter error"));
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration must have 1 input"));
  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("model configuration must have 1 output"));

  common::TritonJson::Value input, output;
  RETURN_IF_ERROR(inputs.IndexAsObject(0, &input));
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));
  const char* input_name;
  size_t input_name_len;
  RETURN_IF_ERROR(input.MemberAsString("name", &input_name, &input_name_len));
  input_name_ = std::string(input_name);

  const char* output_name;
  size_t output_name_len;
  RETURN_IF_ERROR(
      output.MemberAsString("name", &output_name, &output_name_len));
  output_name_ = std::string(output_name);

  std::string input_dtype, output_dtype;
  RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
  RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));
  inp_datatype_ = ModelConfigDataTypeToTritonServerDataType(input_dtype);
  out_datatype_ = ModelConfigDataTypeToTritonServerDataType(output_dtype);

  // dims
  std::vector<int64_t> input_shape;
  std::vector<int64_t> output_shape;
  RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));
  RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));
  output_shape_ = output_shape;
  input_shape_ = input_shape;
  double mean_r, mean_g, mean_b, std_r, std_g, std_b;
  if (ModelConfig().Find("parameters", &parameters)) {
    common::TritonJson::Value mean_R, mean_G, mean_B, std_R, std_G, std_B;
    if (parameters.Find("mean_R", &mean_R)) {
      std::string mean_r_str;
      RETURN_IF_ERROR(
          mean_R.MemberAsString("string_value", &mean_r_str));
      mean_r = std::stod(mean_r_str);
    }
    if (parameters.Find("mean_G", &mean_G)) {
      std::string mean_g_str;
      RETURN_IF_ERROR(
          mean_G.MemberAsString("string_value", &mean_g_str));
      mean_g = std::stod(mean_g_str);
    }
    if (parameters.Find("mean_B", &mean_B)) {
      std::string mean_b_str;
      RETURN_IF_ERROR(
          mean_B.MemberAsString("string_value", &mean_b_str));
      mean_b = std::stod(mean_b_str);
    }
    if (parameters.Find("std_R", &std_R)) {
      std::string std_r_str;
      RETURN_IF_ERROR(
          std_R.MemberAsString("string_value", &std_r_str));
      std_r = std::stod(std_r_str);
    }
    if (parameters.Find("std_G", &std_G)) {
      std::string std_g_str;
      RETURN_IF_ERROR(
          std_G.MemberAsString("string_value", &std_g_str));
      std_g = std::stod(std_g_str);
    }
    if (parameters.Find("std_B", &std_B)) {
      std::string std_b_str;
      RETURN_IF_ERROR(
          std_B.MemberAsString("string_value", &std_b_str));
      std_b = std::stod(std_b_str);
    }
  }
  mean_.push_back(mean_r);
  mean_.push_back(mean_g);
  mean_.push_back(mean_b);
  std_.push_back(std_r);
  std_.push_back(std_g);
  std_.push_back(std_b);
  return nullptr;  // success
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  delete model_state;

  return nullptr;  // success
}

}  
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  ModelState* StateForModel() const { return model_state_; }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance)
      : BackendModelInstance(model_state, triton_model_instance),
        model_state_(model_state)
  {
  }

  ModelState* model_state_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);
  delete instance_state;

  return nullptr;  // success
}

}  

bool isFileExists_access(std::string name) {
    return (access(name.c_str(), F_OK ) != -1 );
}

extern "C" {

TRITONSERVER_Error* FindVaildPath(const char * char_arr, size_t& char_byte_size, bool supports_first_dim_batching, std::vector<std::string>& image_paths){
  if(supports_first_dim_batching){
    size_t ii=0;
    size_t start_idx = 0;
    while(ii < char_byte_size){
      if(char_arr[ii] == '/'){
        start_idx = ii;
        while((ii + 3) < char_byte_size){
          if(std::string(char_arr + ii + 1, char_arr + ii + 4) == "jpg"){
            std::string file_path = std::string(char_arr + start_idx, char_arr + ii + 4);
            if(!isFileExists_access(file_path)){
              return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND,
            "input file path not exists");
            }
            image_paths.push_back(file_path);
            ii = ii + 3;
            break;
          }else{
            ii ++;
          }
        }
      }
      ii ++;
    }
  }else{
    size_t ii =0; 
    for(; ii< char_byte_size; ii++){
      if(char_arr[ii] == '/'){
        break;
      }
    }
    std::string s(char_arr + ii, char_arr + char_byte_size);
    image_paths.push_back(s);
  }
  return nullptr;
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  bool supports_first_dim_batching;

  model_state->SupportsFirstDimBatching(&supports_first_dim_batching);

  size_t dst_c =  model_state->TargetShape()[2];
  size_t dst_h =  model_state->TargetShape()[0];
  size_t dst_w =  model_state->TargetShape()[1];
  size_t output_size = dst_c * dst_h * dst_w;

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);
  for(uint32_t r= 0; r < request_count; ++r){
    TRITONBACKEND_Request* request = requests[r];
    // TRITONBACKEND_Response* response = responses[r];
    TRITONBACKEND_ResponseFactory* factory_ptr;
    RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_ResponseFactoryNew(&factory_ptr, request));
    std::unique_ptr<TRITONBACKEND_ResponseFactory, backend::ResponseFactoryDeleter> factory(factory_ptr);
    // prepare input
    std::vector<std::string> image_paths;
    TRITONBACKEND_Input* in;
    RESPOND_AND_RETURN_IF_ERROR(
      request, TRITONBACKEND_RequestInput(request, "image_path", &in));

    const int64_t* in_shape_arr;
    uint32_t in_dims_count;
    uint64_t in_byte_size;
    RESPOND_AND_RETURN_IF_ERROR(
        request, TRITONBACKEND_InputProperties(
                    in, nullptr, nullptr, &in_shape_arr, &in_dims_count,
                    &in_byte_size, nullptr));
    std::vector<int64_t> in_shape(in_shape_arr, in_shape_arr + in_dims_count);

    const uint32_t element_count = in_byte_size;
    char* in_buffer = new char[element_count];
    RESPOND_AND_RETURN_IF_ERROR(
        request, backend::ReadInputTensor(
                    request, "image_path", in_buffer, &in_byte_size));
    RESPOND_AND_RETURN_IF_ERROR(
        request, FindVaildPath(in_buffer, in_byte_size, supports_first_dim_batching, image_paths));

    for(size_t i=0; i< image_paths.size(); i++){
      LOG_MESSAGE(
        TRITONSERVER_LOG_INFO, image_paths[i].c_str());
    }
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("model ") + model_state->Name() + ": requests count: " +
        std::to_string(request_count) + "batchsize:" + std::to_string(image_paths.size()))
            .c_str());

    // process ouput 
    TRITONBACKEND_Output* out;
    TRITONBACKEND_Response* response;
    responses.push_back(response);
    RESPOND_FACTORY_AND_RETURN_IF_ERROR(
        factory.get(),
        TRITONBACKEND_ResponseNewFromFactory(&response, factory.get()));

    float* data = (float*)malloc(sizeof(float) * output_size * image_paths.size());
    const int64_t* out_shape;
    size_t dims;
    if(supports_first_dim_batching){
      out_shape = new int64_t[]{int64_t(image_paths.size()), int64_t(dst_h), int64_t(dst_w), int64_t(dst_c)};
      dims = model_state->TargetShape().size() + 1;
    }else{
      out_shape = new int64_t[]{int64_t(dst_h), int64_t(dst_w), int64_t(dst_c)};
      dims = model_state->TargetShape().size();
    }

    RESPOND_FACTORY_AND_RETURN_IF_ERROR(
        factory.get(), TRITONBACKEND_ResponseOutput(
                           response, &out, "image", TRITONSERVER_TYPE_FP32,
                           out_shape, dims));
    delete []out_shape;
    void* out_buffer;
    TRITONSERVER_MemoryType out_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t out_memory_type_id = 0;

    RESPOND_FACTORY_AND_RETURN_IF_ERROR(
        factory.get(), TRITONBACKEND_OutputBuffer(
                           out, &out_buffer, sizeof(float) * output_size * image_paths.size(), &out_memory_type,
                           &out_memory_type_id));

    if (out_memory_type == TRITONSERVER_MEMORY_GPU) {
      RESPOND_FACTORY_AND_RETURN_IF_ERROR(
          factory.get(),
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              "failed to create OUT output buffer in CPU memory"));
    }

    for(size_t ii=0; ii < image_paths.size(); ii++){
      std::string image_path = image_paths[ii];
      cv::Mat img = cv::imread(image_path);
      cv::resize(img, img, cv::Size(dst_w, dst_h));
      cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
      for(int row=0; row < img.rows; row ++){
        for(int col=0; col < img.cols; col ++){
          cv::Vec3b val = img.at<cv::Vec3b>(row, col);
          float r = (val[0] - model_state->NormMean().x) / model_state->NormStd().x;
          float g = (val[1] - model_state->NormMean().y) / model_state->NormStd().y;
          float b = (val[2] - model_state->NormMean().z) / model_state->NormStd().z;
          data[ii * output_size + row * dst_w *dst_c + col * dst_c] = r;
          data[ii * output_size + row * dst_w *dst_c + col * dst_c + 1] = g;
          data[ii * output_size + row * dst_w *dst_c + col * dst_c + 2] = b;
        }
      }
    }
    memcpy(out_buffer, data, sizeof(float) * output_size * image_paths.size());
    LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL /* flags */, nullptr /* success */),
          "failed sending response");
  }
  
  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);
  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

#ifdef TRITON_ENABLE_STATS
  size_t total_batch_size = 0;
  if (!supports_first_dim_batching) {
    total_batch_size = request_count;
  } else {
    for (uint32_t r = 0; r < request_count; ++r) {
      auto& request = requests[r];
      TRITONBACKEND_Input* input = nullptr;
      LOG_IF_ERROR(
          TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input),
          "failed getting request input");
      if (input != nullptr) {
        const int64_t* shape = nullptr;
        LOG_IF_ERROR(
            TRITONBACKEND_InputProperties(
                input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr),
            "failed getting input properties");
        if (shape != nullptr) {
          total_batch_size += shape[0];
        }
      }
    }
  }
#else
  (void)exec_start_ns;
  (void)exec_end_ns;
  (void)compute_start_ns;
  (void)compute_end_ns;
#endif  // TRITON_ENABLE_STATS

  // Report statistics for each request, and then release the request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];

#ifdef TRITON_ENABLE_STATS
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_state->TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");
#endif  // TRITON_ENABLE_STAT
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

#ifdef TRITON_ENABLE_STATS
  // Report batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_state->TritonModelInstance(), total_batch_size,
          exec_start_ns, compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");
#endif  // TRITON_ENABLE_STATS

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::recommended
