//
// Created by sun on 2021/11/18.
//

#ifndef AUTOFUSION_COMPUTATION_H
#define AUTOFUSION_COMPUTATION_H

#include<string>
#include<vector>
#include<unordered_map>
#include<map>
#include<set>
#include<Formula.h>

#include "Affine.h"
#include"Polynomial.h"

#define MAX_INPUT 2

enum TensorType{
    INPUT , //means not constant at compile time
    WEIGHT ,//means constant at compile time
    MIX,//combine input and weight

};


enum SupportedDataType{
    SDT_FLOAT,
    SDT_INT,
    SDT_UINT,
    SDT_INVALID,
};
const vector<string> supported_data_type_string={
        "float","int","uint","var"
};
enum SubOpType{
    DATA,
    ADD,
    MUL,
    ASSIGN,
    SUM,
    COND,
    NOP,
    NEG,
    MAX,
    SMAX,
    LOG,
    EXP,
    REC,
    SQRT,
    TRANSFORM,
    RESHAPE,
};
enum SubOpMetaType{
    COMPUTATION,
    MEMORY,
    REDUCE,
    NONLINEAR,
    LINEAR,
    ELEMENT,
    PSEUDO,
};
union var32{
    int int_value;
    float float_value;
    uint uint_value;
};
static std::string sub_op_type_string[]={
        "data",
        "add",
        "multiply",
        "assign",
        "reduce_sum",
        "cond",
        "nop",
        "neg",
        "reduce_max",
        "smax",
        "log",
        "exp",
        "reciprocal",
        "sqrt",
        "transform",
        "reshape",
};
static vector<string> data_type_string={
        "float32",
        "float64",
        "float16",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "bool"

};
using namespace std;
class Computation;
/*
 * Presburger set:
 * ...
 *
 * About DataType:
 * since this is a compiler system and it not targeting specific hardware
 * and we only use DataType to  pre-compute weights,so we limit this system's DataType to three data type:
 * float-- we using float32 to preprocess weights
 * signed integer-- we using int32 to preprocess weights
 * unsigned integer-- we using uint32 to preprocess weights
 * bool -- bool(is that useful? do not support it)
*/
class Tensor{
public:
    string name;
    vector<Variable> shape;//todo use unordered_set instead of vector
    vector<Variable> original_shape; //for reshape,record the original shape
    int index;
    map<int,int> related_idx;
    SubOpType sub_op_type{DATA};
    bool is_used_for_te{false}; //for codegen_te
    TensorType tensorType{WEIGHT};
    DataType dataType{DT_FLOAT};
    set<int> group_id;//todo use int instead of set<int>
    bool is_boundary{false};
    string serialization;// for combining two or more compute
    vector<string> attribute;//for combining two or more compute
    ////using void* rather than var32* to keep expandability
    void* data_ptr{nullptr};
    Tensor& operator=(const Tensor& t);
    bool operator!=(const Tensor& t)const ;
    bool operator==(const Tensor& t)const;
    [[nodiscard]] bool is_same_initial_weight(const Tensor& t)const;
    [[nodiscard]] int get_length() const;
    void reshape(const vector<Variable>& new_shape);
    void reshape(const vector<int>& new_shape);

    Tensor();
    Tensor(const Tensor& t);
    Tensor(const vector<Variable>& _shape, TensorType _type, string name);
    Tensor(const vector<Variable>& _shape, TensorType _type, void* _ptr, string name);

    Tensor(const vector<Variable>& _shape, TensorType _type, DataType _dataType, void* _ptr, string name);// priority call

    Tensor(const vector<int>& _shape, TensorType _type, string name);//dispose
    Tensor(const vector<int>& _shape, TensorType _type, void* _ptr, string name);//dispose
    string to_string(bool concrete=false,bool c = false,bool title=true,bool bracket=true,bool has_brace=true,bool is_original=false);
    string to_placeholder();
    void print_data(ostream& out);
    static Tensor zero;
    static Tensor one;
    static SubOpMetaType convert_sub_op_type(SubOpType _op);

    static SupportedDataType convert_data_type(DataType );
private:


};
using TensorHandle = Tensor*;
//todo refactor: new out Tensor at Base class ,no need to pass a TensorHandle
class Computation{
public:
    TensorHandle out;
    TensorHandle in[MAX_INPUT];
    int num_in;
    SubOpType op;
    int num_consumer{};
    bool is_boundary{false};

    Computation& operator=(const Computation& c);
    Computation();
    Computation(TensorHandle , SubOpType, TensorHandle , TensorHandle , int _idx);
    Computation(TensorHandle , SubOpType, TensorHandle , int _idx);
    virtual string op_to_string() ;
    virtual ~Computation();
    virtual void compute();
    virtual string to_compute();
    virtual void set_compute();
    virtual string to_c_statement();
    //wrapper function
    static void* allocate(SupportedDataType t, const vector<Variable>& shape,float value=0);
    static void* allocate_random(SupportedDataType t, const vector<Variable>& shape);
    //helper function
    void unary_shape_inference();
    void binary_shape_inference();
};
class Sum:public Computation{
public:
    vector<Variable> reduce_axis; //make sure reduce_axis is not empty
    vector<int> pads;
    vector<Variable> pad_axis;
    vector<Variable> old_axis;//parts of reduce_axis without channel
    //assuming only substitute one variable
    pair<Variable,Variable> subst;// now,original
    // for combine two sum with same stride (if we can combine two sum with different stride then eliminate it)
    vector<int> stride{1,1}; // make sure stride >0
    Sum(TensorHandle out,TensorHandle in,const vector<Variable>& _v,int idx,const vector<int>& stride={1,1},bool bound=true);
    string op_to_string() override;
    void shape_inference();
    void compute() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
    string to_c_statement_initial();
    vector<string> to_reduce_axis();
    vector<Variable> get_original_shape();
    bool is_stride_equals(int value=1);
};

class Max:public Computation{
public:
    vector<Variable> reduce_axis;
    // in theory,if we combine two max we need to make sure stride is same,leave it a todo
    Max(TensorHandle   out,TensorHandle   in,const vector<Variable>& _v,int idx);
    void compute() override;
    string op_to_string() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
    void shape_inference();
    string to_c_statement_initial();
    vector<string> to_reduce_axis();
};

class Mul:public Computation{
public:
    Mul(TensorHandle  , TensorHandle   in1,TensorHandle   in2,int ,const vector<Variable>& simplify={});
    string op_to_string() override;
    void compute() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
    vector<Variable> simplify_const_tensor_indices;
};
class Add:public Computation{
public:
    Add(TensorHandle  , TensorHandle   in1,TensorHandle   in2,int );
    void compute() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
};

class SMax:public Computation{
public:
    SMax(TensorHandle  ,TensorHandle   in1,TensorHandle  in2,int );
    void compute() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
};

class Assign:public Computation{
public:
//    bool is_pad;
//    bool is_dim_expansion;
//    vector<int> pads;//works when isPad == true
//    vector<Variable> old_axis;//pass from top computation
//    vector<Variable> pad_axis;//pass from top computation
    map<Variable,Affine> mapping;
    vector<Affine> lambda;
    Formula* formula;
    // no shape inference , need user to specify it
    Assign(TensorHandle  out,TensorHandle  in,const vector<Variable>& out_shape,int idx,const vector<Affine>& in_lambda); //autogen formula
    Assign(TensorHandle  out,TensorHandle  in,const vector<Variable>& out_shape,int idx,const vector<Affine>& in_lambda,Formula* ptr); //set formula or set nullptr to formula
    Assign(TensorHandle out,TensorHandle in,const vector<Variable>& out_shape,int idx,const map<Variable,Affine>& mapping);//autogen formula
    Assign(TensorHandle out,TensorHandle in,const vector<Variable>& out_shape,int idx,const map<Variable,Affine>& mapping,Formula* ptr);//set formula or set nullptr to formula
    // only for pad mode
    //Assign(TensorHandle  ,TensorHandle   in,const vector<Variable>& out_shape,const vector<Variable>& pad_axis,const vector<int>& pads,int idx,const vector<Variable>& old);
    string op_to_string() override;
    void compute() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
    bool is_stride_equals(int value=1);

private:
    //helper function
};
/*
 * there is no NOP in the optimized code
 */
class Nop:public Computation{
public:

    Nop(TensorHandle  _out,TensorHandle  in,int idx);
};
class Reshape:public Computation{
public:
    Reshape(TensorHandle  _out,TensorHandle  in,int idx,const vector<int>& new_shape);
    vector<int> new_shape;
    string to_compute() override;
};
class Cond:public Computation{
public:
//    vector<Expr> cond_expr;
//    Cond(TensorHandle  , vector<Expr>& _expr,TensorHandle ,TensorHandle ,const vector<Variable>& out_shape,int idx,
//         const vector<Expr>& in1_expr={},const vector<Expr>& in2_expr={});

    Variable in0_axis;
    Variable in1_axis;
    Variable new_axis;
    Cond(TensorHandle,TensorHandle,TensorHandle,const Variable&,const Variable & ,const Variable&,int idx);
    static bool check_cond(const vector<Variable>& shape1,const vector<Variable>& shape2);
    static bool check_cond(const vector<Variable>& shape1,const vector<Variable>& shape2,const Variable& axis0,const Variable& axis1);
    string op_to_string() override;
    void compute() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
};

class Neg:public Computation{
public:

    Neg(TensorHandle  _out, TensorHandle  in,int idx);
    void compute() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
};

class Log:public Computation{
public:

    Log(TensorHandle  _out, TensorHandle  in,int idx);
    void compute() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
};
class Exp:public Computation{
public:

    Exp(TensorHandle  _out, TensorHandle  in,int idx);
    void compute() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
};

class Rec:public Computation{
public:

    Rec(TensorHandle  _out, TensorHandle  in,int idx);
    void compute() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
};
class Sqrt:public Computation{
public:

    Sqrt(TensorHandle  _out,TensorHandle  in,int idx);
    void compute() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
};

class Transform:public Computation{
public:
    map<Intermedia,float> variable_table;
    int count{0}; //how many elements used
    set<Function> function{};
    Transform(TensorHandle  _out,TensorHandle  in,const vector<Variable>& out_shape,int idx,int count=0);
    //todo
    void compute() override;
    string to_compute() override;
    void set_compute() override;
    string to_c_statement() override;
};

//class Div:public Computation{
//public:
//    float divider;
//    Div(TensorHandle  _out,TensorHandle  in,float _d,int idx);
//    void shape_inference();
//    string op_to_string() override;
//
//};


#endif //AUTOFUSION_COMPUTATION_H
