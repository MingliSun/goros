//
// Created by sun on 2021/11/18.
//

#ifndef AUTOFUSION_OPS_H
#define AUTOFUSION_OPS_H
#include"Computation.h"
#include"Parser.h"
#include"Polynomial.h"
#include<list>

#define MAX_NUM_INPUTS 6
#define MAX_NUM_OUTPUTS 6
#define MAX_NUM_COMPUTATION 1000
#define MAX_NUM_FUSED_COMPUTATION 10

enum OpType {
    OP_UPSAMPLING,
    OP_CONVOLUTION,
    OP_DROPOUT,
    OP_NONLINEAR,
    OP_POOLING_MAX,
    OP_POOLING_AVG,
    OP_RELU,
    OP_SIGMOID,
    OP_TANH,
    OP_BATCH_NORM,
    OP_BATCH_FLATTEN,
    OP_CONCAT,
    OP_SPLIT,
    OP_RESHAPE,
    OP_TRANSPOSE,
    OP_EW_ADD,
    OP_EW_MUL,
    OP_MATMUL,
    OP_BIAS_ADD,
    OP_BIAS_MUL,
    OP_BROADCAST_ADD,
    OP_BROADCAST_MUL,
    OP_SOFTMAX,
    OP_TRANSPOSED_CONVOLUTION,
    OP_SQUEEZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Squeeze
    OP_UNSQUEEZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Unsqueeze
    OP_EW_SUB, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sub
    OP_EW_DIV, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Div
    OP_EW_EQUAL, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Equal
    OP_EW_GREATER, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Greater
    OP_EW_LESS, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Less
    OP_EW_MAX, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Max
    OP_EW_MIN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Min
    OP_REDUCE_ARGMAX, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMax
    OP_REDUCE_ARGMIN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ArgMin
    OP_REDUCE_MAX, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMax
    OP_REDUCE_MEAN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMean
    OP_REDUCE_MIN, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceMin
    OP_REDUCE_PROD, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceProd
    OP_REDUCE_SUM, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#ReduceSum
    OP_PAD, //https://github.com/dmlc/tvm/blob/master/topi/python/topi/nn/pad.py
    OP_SHAPE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Shape
    OP_SIZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Size
    OP_TOPK, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#TopK
    OP_WHERE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Where
    OP_CEIL, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Ceil
    OP_CAST, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Cast
    OP_EXP, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Exp
    OP_ROUND, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Round
    OP_LOG, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Log
    OP_LOGICAL_NOT, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Not
    OP_SQRT, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sqrt
    OP_LEAKYRELU,
    OP_SLICE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Slice
    OP_RESIZE, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#Resize
    OP_PRELU, //https://github.com/onnx/onnx/blob/master/docs/Operators.md#PRelu
};

enum PaddingMode{
    PD_MODE_VALID,
    PD_MODE_SAME,
};
class Graph;
class OpBase {
public:
    OpBase(TensorHandle  input,  OpType _type,Graph* _graph);//1
    OpBase(TensorHandle  input0, TensorHandle  input1,
            OpType _type,Graph* _graph);//2
    OpBase(TensorHandle  input0, TensorHandle  input1, TensorHandle  input2,
            OpType _type,Graph* _graph);//3
    OpBase(TensorHandle  input0, TensorHandle  input1, TensorHandle  input2,
           TensorHandle  input3,  OpType _type,Graph* _graph);//4
    OpBase(TensorHandle  input0, TensorHandle  input1,
           TensorHandle  input2, TensorHandle  input3,
           TensorHandle  input4,  OpType _type,Graph* _graph);//5
    OpBase(int n, TensorHandle* inputs,  OpType _type,Graph* _graph);//n
    OpBase(vector<TensorHandle> inputs,OpType _type,Graph* _graph);

public:
    TensorHandle inputs[MAX_NUM_INPUTS], outputs[MAX_NUM_OUTPUTS];
    int numInputs, numOutputs{1};
    OpType type;
    Graph* graph;

};

//todo dynamic shape and operator-level cse
class FusedOperator;
class Graph{
public:
    Graph();
    TensorHandle new_input(const vector<int>& shape,DataType dataType=DT_FLOAT,string name="");
    TensorHandle new_weight(const vector<int>& shape,DataType dataType=DT_FLOAT,string name="",bool depthwise=false);
    //new bias weight
    TensorHandle new_weight(int shape,int axis=1,DataType dataType=DT_FLOAT,string name="");

    TensorHandle conv2d(TensorHandle  _input,TensorHandle  _weight,int _strideH, int _strideW, PaddingMode _padding);//todo may rebuild it with GroupConvolution
    TensorHandle conv2d_group(TensorHandle  _input,TensorHandle  _weight,int _strideH, int _strideW, PaddingMode _padding);//infer group
    TensorHandle conv3d(TensorHandle  _input,TensorHandle  _weight,int _strideD,int _strideH, int _strideW, PaddingMode _padding);
    TensorHandle add(TensorHandle  _left,TensorHandle  _right);
    TensorHandle multiply(TensorHandle  _left, TensorHandle  _right);
    TensorHandle element(TensorHandle  _left,TensorHandle  _right,OpType _type); //todo will remove it at next time
    TensorHandle batch_norm(TensorHandle  _input, TensorHandle  _scale, TensorHandle  _bias, TensorHandle  _mean, TensorHandle  _var,
                            float _epsilon);
    TensorHandle relu(TensorHandle _input);
    TensorHandle leaky_relu(TensorHandle _input,float alpha=0.1);
    TensorHandle max_pool2d(TensorHandle _input, int _kernelH, int _kernelW, int strideH, int strideW, PaddingMode _padding);
    TensorHandle global_max_pool2d(TensorHandle _input);
    TensorHandle avg_pool2d(TensorHandle _input, int _kernelH, int _kernelW, int strideH, int strideW, PaddingMode _padding);
    TensorHandle global_avg_pool2d(TensorHandle _input);
    TensorHandle dropout(TensorHandle _input,float rate);
    TensorHandle conv2d_transpose(TensorHandle  _input,TensorHandle  _weight,int _strideH, int _strideW,const vector<int>& out_padding );
    TensorHandle concat(const vector<TensorHandle>& input, int axis);
    TensorHandle upsampling(TensorHandle _input, const base& scale_h, const base& scale_w,const string& method="nearest_neighbor");//interpolate
    TensorHandle sigmoid(TensorHandle _input);
    TensorHandle tanh(TensorHandle _input);
    TensorHandle log(TensorHandle _input);
    TensorHandle batch_flatten(TensorHandle _input);
    TensorHandle reshape(TensorHandle _input,const vector<int>& new_shape);
    TensorHandle dense(TensorHandle _input,TensorHandle _weight);
    TensorHandle bias_add(TensorHandle _input,TensorHandle _bias);
    TensorHandle softmax(TensorHandle _input,int axis=-1);
    TensorHandle yolo_reorg(TensorHandle _input,int stride=2);
    TensorHandle pad(TensorHandle _input,const vector<int>& pad_axis,const vector<int>& pad_value);
    TensorHandle slice(TensorHandle _input,const vector<int>& starts,const vector<int>& ends,const vector<int>& axes,const vector<int>& steps);
    TensorHandle transpose(TensorHandle _input,const vector<int>& perm={});
    void split(TensorHandle input,int indices,int axis,vector<TensorHandle>& output);
    void split(TensorHandle input,const vector<int>& sections,int axis, vector<TensorHandle>& output);

    void function(const vector<TensorHandle>& net);
    void optimize();
    int num_cpt{0};
    int frontIdx{-1}; //todo change data structure of frontIdx and remove it
    Computation* computations[MAX_NUM_COMPUTATION]{};
    void push(Computation* _computation);
    string to_string();
    static map<int,Variable> weight_oc;
    static map<int,Variable> channel;
    static map<int,Variable> weight_kd;//for 3d
    static map<int,Variable> weight_kh;
    static map<int,Variable> weight_kw;
    static map<int,Variable> input_n;
    static map<int,Variable> input_d;//for 3d
    static map<int,Variable> input_h;
    static map<int,Variable> input_w;
    static map<int,Variable> other;
    set<Variable> concat_axis;
    vector<TensorHandle> inputs;
    vector<TensorHandle> outputs;

    vector<FusedOperator> fused_operators;
    void codegen_te(const string& filename,int trial=500,bool resume=false,const string& target="cuda",const string& host="llvm");
    void codegen_c(const string& filename);
    void codegen_dot(const string& filename);
    void print_weights(const string& filename);

    //helper function
    Compare is_single_concat_axis(const vector<Variable>& var0s,const vector<Variable>& var1s,Variable& v0,Variable& v1);
    void adjust_sum(Sum* sum);
    static Variable get_variable(map<int,Variable>& m,int value,int row,int column);
    static Variable get_variable(int axis,int value,int size);//wrapper of top
    static vector<Variable> get_variable(const vector<int>& shape,bool input=true,bool depthwise=false); //get input variable

private:
    //helper function
    static void float2fraction(float target,int & numerator,int& denominator);
    string set_to_string(const set<int>& s);
    static bool is_element_in_set(const set<int>& s,int element);
};
class FusedOperator{
public:
    int id{-1};
    string name;
    FusedOperator();
    vector<TensorHandle> signature;
    vector<TensorHandle> intermediate_variable;
    vector<Computation*> computations;
    vector<string> source_te;
    vector<vector<Variable>> c_loops;
    vector<vector<string>> c_statements;
    vector<string> source_dot;
    void push(Computation* _computation);
    void generate_te();
    void generate_dot();
    void generate_c();

    //helper function
private:
    static bool is_contain(const vector<Variable>& ,const vector<Variable>& small);
};

class Convolution: public OpBase{
public:

    Convolution(TensorHandle _input, TensorHandle _weight, const vector<int>& stride, PaddingMode _padding, Graph*g);
    ~Convolution() = default;
    vector<int> stride;
    PaddingMode padding;
};

class Element:public OpBase{
public:
    Element(TensorHandle  left,TensorHandle  right,OpType _type,Graph*g);
    ~Element();
    //static bool is_broadcastable(TensorHandle t1,TensorHandle t2);
};

class BatchNorm:public OpBase{
public:
    BatchNorm(TensorHandle  _input, TensorHandle  _scale,
              TensorHandle  _bias, TensorHandle  _mean, TensorHandle  _var,
              float _epsilon,Graph*g);
    ~BatchNorm();

    float epsilon;
};

class MaxPooling:public OpBase{
public:
//    MaxPooling(TensorHandle _input,int _kernelH,int _kernelW,int strideH,int strideW,PaddingMode _padding,Graph* g);
    MaxPooling(TensorHandle _input,const vector<int>& kernel,const vector<int>& stride,PaddingMode _padding,Graph* g);
    ~MaxPooling() = default;
    vector<int> stride;
    vector<int> kernel;
    PaddingMode padding;
};
class Dropout:public OpBase{
public:
    Dropout(TensorHandle _input,float _level,Graph* g);
    ~Dropout() = default;
    float prob;
};
class TransposedConvolution: public OpBase{
public:
    TransposedConvolution(TensorHandle _input, TensorHandle _weight,const vector<int>& stride, const vector<int>& out_padding, Graph*g);
    ~TransposedConvolution() = default;
};

class GroupConvolution: public OpBase{
public:
    GroupConvolution(TensorHandle _input, TensorHandle _weight, const vector<int>& stride, PaddingMode _padding, Graph*g);
    ~GroupConvolution() = default;
};
class Concatenate:public OpBase{
public:
    //Concatenate();
    Concatenate(const vector<TensorHandle>& inputs,int axis,Graph* g);
    ~Concatenate() = default;
    int axis;
private:
    [[nodiscard]] bool check(const vector<Variable>& shape1,const vector<Variable>& shape2) const;
};
class NearestNeighbor: public OpBase{
public:
    NearestNeighbor(TensorHandle _input, const vector<int>& scale, Graph* g);
    NearestNeighbor(TensorHandle _input,const vector<int>& numerator,const vector<int>& denominator,Graph* g); //for float scale
    ~NearestNeighbor() = default;
};
class Sigmoid:public OpBase{
public:
    Sigmoid(TensorHandle _input,Graph* g);
    ~Sigmoid() = default;
};
class AvgPooling:public OpBase{
public:

    AvgPooling(TensorHandle _input,const vector<int>& kernel,const vector<int>& stride,PaddingMode _padding,Graph* g);
    ~AvgPooling()=default;
    vector<int> stride;
    vector<int> kernel;
    PaddingMode padding;
};
class BatchFlatten:public OpBase{
public:
    explicit BatchFlatten(TensorHandle _input,Graph* g);
    ~BatchFlatten() = default;
};
class Dense: public OpBase{
public:
    Dense(TensorHandle _input, TensorHandle _weight, Graph* g);
    ~Dense() = default;
};
class Softmax:public OpBase{
public:
    Softmax(TensorHandle _input,const Variable& v,Graph* g);
    ~Softmax() = default;
};
class YoloReorg:public OpBase{
public:
    YoloReorg(TensorHandle _input,int stride,Graph* g);
    ~YoloReorg() = default;
};
class Split:public OpBase{
public:
    vector<int> sections;
    Split(TensorHandle _input,const vector<int>& sections,int axis,Graph* g);
    ~Split() = default;

private:
    //wrapper of Graph::get_variable()
};
class Tanh:public OpBase{
public:
    Tanh(TensorHandle _input,Graph* g);
    ~Tanh() = default;
};
class Pad: public OpBase{
public:
    Pad(TensorHandle _input,const vector<int>& pad_axis,const vector<int>& pad_value,Graph* g);
    ~Pad() =default;
};
class Slice:public OpBase{
public:
    Slice(TensorHandle _input,const vector<int>& starts,const vector<int>& ends,const vector<int>& axes,const vector<int>& steps,Graph* g);
    ~Slice() = default;
};
class Winograd:public OpBase{
public:
    Winograd(TensorHandle _input, TensorHandle _weight, int tile_size,PaddingMode _padding, Graph*g);
    ~Winograd() = default;
};
class Bilinear:public OpBase{
public:
    Bilinear(TensorHandle _input,float h_scale,float w_scale, Graph* g);
    ~Bilinear()= default;
};
class Bicubic:public OpBase{
    Bicubic(TensorHandle _input,float h_scale,float w_scale, Graph* g);
    ~Bicubic() = default;
};
class Trilinear:public OpBase{
    Trilinear(TensorHandle _input,float scale_d,float scale_h,float scale_w, Graph* g);
    ~Trilinear() = default;
};
class Transpose:public OpBase{
public:
    Transpose(TensorHandle _input,const vector<int>& perm,Graph* g);
    ~Transpose() = default;
};
#endif //AUTOFUSION_OPS_H
