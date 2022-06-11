//
// Created by sun on 2021/11/18.
//

#include"Computation.h"
#include"Producer.h"
#include"Pattern.h"
#include<cassert>
#include <utility>
#include<cmath>
#include<iostream>
#include<stack>
Tensor Tensor::zero = Tensor({},WEIGHT,DT_FLOAT,new float {0},"zero");
Tensor Tensor::one = Tensor({},WEIGHT,DT_FLOAT,new float {1},"one");
const float infinite = float(1.0)/float(0.0);

Computation::Computation(TensorHandle _out, SubOpType t, TensorHandle _in0, TensorHandle _in1, int idx): op(t), num_in(2), out(_out){
    _out->sub_op_type = t;
    _out->index = idx;
    this->in[0] = _in0;
    this->in[1] = _in1;


    _in0->related_idx.insert(make_pair(idx,0));
    _in1->related_idx.insert(make_pair(idx,1));

    if(_in0->tensorType==WEIGHT&&_in1->tensorType==WEIGHT){
        _out->tensorType = WEIGHT;
    }else if(_in0->tensorType==INPUT&&_in1->tensorType==INPUT){
        _out->tensorType = INPUT;
    }else if(_in0->tensorType==MIX||_in1->tensorType==MIX||(_in0->tensorType==INPUT&&_in1->tensorType==WEIGHT)||(_in0->tensorType==WEIGHT&&_in1->tensorType==INPUT)){
        _out->tensorType = MIX;
    }
    assert(_in0->dataType==_in1->dataType);
    _out->dataType = _in0->dataType;
}

Computation::Computation():op(DATA),num_in(0) {}

Computation &Computation::operator=(const Computation &c) {
    if(this==&c) return *this;
    this->op = c.op;
    this->num_in = c.num_in;
    this->out = c.out;
    for(int i=0;i<num_in;i++){
        this->in[i] = c.in[i];
    }
    return *this;
}

Computation::Computation(TensorHandle _out , SubOpType t, TensorHandle _in , int idx) {
    _out->sub_op_type = t;
    _out->index = idx;
    this->op = t;
    this->num_in = 1;
    this->in[0] = _in;
    this->in[1]=nullptr;
    this->out = _out;

    _out->tensorType = _in->tensorType;

    //_out->setShape(_in->shape);

    _in->related_idx.insert(make_pair(idx,0));
    _out->dataType = _in->dataType;
}

string Computation::op_to_string() {
    return sub_op_type_string[op];
}

Computation::~Computation() = default;

void Computation::compute() {

}

string Computation::to_compute() {
    return std::__cxx11::string("te.compute(...)");
}

string Computation::to_c_statement() {
    return std::__cxx11::string("statement...");
}

void *Computation::allocate(SupportedDataType t, const vector<Variable>& shape,float value) {
    void* result;
    switch (t) {
        case SDT_FLOAT:
            result = Producer::allocate<float>(shape,value);
            break;
        case SDT_INT:
            result = Producer::allocate<int>(shape,value);
            break;
        case SDT_UINT:
            result = Producer::allocate<uint>(shape,value);
            break;
        case SDT_INVALID:
            result = nullptr;
            break;
    }
    return result;
}

void *Computation::allocate_random(SupportedDataType t, const vector<Variable> &shape) {
    //allocate_random to reduce time
    void* result;
    switch (t) {
        case SDT_FLOAT:
            result = Producer::allocate<float>(shape);
            break;
        case SDT_INT:
            result = Producer::allocate<int>(shape);
            break;
        case SDT_UINT:
            result = Producer::allocate<uint>(shape);
            break;
        case SDT_INVALID:
            result = nullptr;
            break;
    }
    return result;
}

void Computation::unary_shape_inference() {
    out->shape = in[0]->shape;
}

void Computation::binary_shape_inference() {
    vector<Variable> v;
    // special case : is not overlap
    if(!PatternMatch::is_overlap(in[0]->shape,in[1]->shape)){
        auto i = in[0]->shape.begin();
        auto j = in[1]->shape.begin();
        while(i!=in[0]->shape.end()||j!=in[1]->shape.end()){
            if(i!=in[0]->shape.end()) v.push_back(*i++);
            if(j!=in[1]->shape.end()) v.push_back(*j++);
        }
        out->shape = std::move(v);
        return;
    }
    // overlapped
    int secondIdx = 0;
    for(auto & i : in[0]->shape){
        bool found = false;
        int j;
        for(auto k=v.begin();k!=v.end();k++){
            if(i==*k){// has been added to v
                v.erase(k);
                break;
            }
        }
        for( j=secondIdx;j<in[1]->shape.size();j++){
            if(i==in[1]->shape[j]){found = true;break;}
        }
        if(!found){
            v.push_back(i);
        }else{
            for(int k=secondIdx;k<=j;k++) {v.push_back(in[1]->shape[k]);}
            secondIdx = j+1;
        }
    }
    for( int j=secondIdx;j<in[1]->shape.size();j++){
        v.push_back(in[1]->shape[j]);
    }
    out->shape = std::move(v);
}

void Computation::set_compute() {

}


Tensor::Tensor(const vector<Variable>& _shape, TensorType _type, string name): index(-1), tensorType(_type), shape(_shape), sub_op_type(DATA) {
    //this->num_dim = _shape.size();
    if(name.empty()){
        this->name = std::move(Producer::get_unique_tensor_name());
    }else{
        this->name = std::move(name);
    }
}

Tensor&  Tensor::operator=(const Tensor&  t) {
    if  (this==&t) return *this;
    this->sub_op_type = t.sub_op_type;
    //this->num_dim = t.num_dim;
    this->name = t.name;
    this->shape = t.shape;
    this->index = t.index;
    this->tensorType = t.tensorType;
    this->related_idx = t.related_idx;
    this->group_id = t.group_id;
    return *this;
}
bool Tensor::operator!=(const Tensor& t)const {
    return !(*this==t);
//    if(this->sub_op_type != t.sub_op_type) return true;
//    if(this->meta_type!=t.meta_type) return true;
//    if(this->tensorType!=t.tensorType) return true;
//    if(this->dataType!=t.dataType) return true;
//    if(this->shape!=t.shape) return true;
//    return false;
}
bool Tensor::operator==(const Tensor&  t)const {
    // name only used to print,not used to compare
    if(this->sub_op_type != t.sub_op_type) return false;
    if(this->tensorType!=t.tensorType) return false;
    if(this->dataType!=t.dataType) return false;
    if(this->shape!=t.shape) return false;
    return true;
}

//conservative set tensorType = INPUT
Tensor::Tensor(): index(-1) , tensorType(INPUT), dataType(DT_FLOAT), data_ptr(nullptr), sub_op_type(DATA){
    this->name = std::move(Producer::get_unique_tensor_name());
}

Tensor::Tensor(const Tensor&   t) {//do not copy data_ptr yet
    if  (this==&t) return ;
    this->sub_op_type = t.sub_op_type;
    this->shape = t.shape;
    this->index = t.index;
    this->name = t.name;
    this->tensorType = t.tensorType;
    this->related_idx = t.related_idx;
    this->dataType = t.dataType;
}

string Tensor::to_string(bool concrete,bool c,bool title,bool bracket,bool has_brace,bool is_original) {
    string comma;
    string lbracket,rbracket;
    if(bracket) lbracket="[",rbracket="]";
    else lbracket="(",rbracket=")";
    if(c) comma = "][";
    else comma = ",";
    vector<Variable> process_shape;
    if(is_original&& !original_shape.empty()) process_shape = original_shape;
    else process_shape = shape;
    string result;
    if(title) result+=name;
    if(has_brace) result +=lbracket;
    for(int i=0; i < process_shape.size(); i++){
        if(concrete) result+=process_shape[i].upper_bound.to_string();
        else result+=process_shape[i].name;
        if(!bracket&&process_shape.size()==1) result+=comma; // (m,)
        if(i != process_shape.size() - 1) result+= comma;
    }
    if(!concrete&&has_brace&&process_shape.empty()) result+= ::to_string(0);
    if(has_brace) result+=rbracket;
    return result;
}

//bool Tensor::is_equal_general(const Tensor&a,const Tensor& b) {
//    if(a.sub_op_type != b.sub_op_type) return false;
//    if(a.shape.size() != b.shape.size()) return false;
//    for(int i=0;i<a.shape.size(); i++){
//        if(a.shape[i]!=b.shape[i]) return false;
//    }
//    return true;
//}


Tensor::Tensor(const vector<int> &_shape, TensorType _type, string name) : index(-1), tensorType(_type), sub_op_type(DATA), name(std::move(name)){
    for(int i=0;i<shape.size();i++){
        shape.emplace_back(Producer::get_unique_variable_name(),_shape[i]);
    }
}

Tensor::Tensor(const vector<int> &_shape, TensorType _type, void *_ptr, string name)
                : index(-1), tensorType(_type), sub_op_type(DATA), data_ptr(_ptr), name(std::move(name)){
    for(int i=0;i<shape.size();i++){
        shape.emplace_back(Producer::get_unique_variable_name(),_shape[i]);
    }
}

Tensor::Tensor(const vector<Variable> &_shape, TensorType _type, void *_ptr, string name): index(-1),
                                                                                           data_ptr(_ptr), tensorType(_type), sub_op_type(DATA), shape(_shape) {
    //this->num_dim = _shape.size();
    if(name.empty())   this->name = std::move(Producer::get_unique_tensor_name());
    else this->name = std::move(name);
}

string Tensor::to_placeholder() {
    string result = this->name+" = te.placeholder(";

    result+=to_string(true,false,false,false,true);

    result+=", name='"+this->name+"', dtype='"+data_type_string[this->dataType]+"')";
    return result;
}

//string Tensor::to_concrete_shape_string(bool ctype) {
//    string comma;
//    if(ctype) comma = "][";
//    else comma = ",";
//    string result = name;
//    result +="[";
//    for(int i=0; i < shape.size(); i++){
//        result+=::to_string(shape[i].get_upper_bound());
//        if(i != shape.size() - 1) result+= comma;
//    }
//    result+="]";
//    return result;
//}

Tensor::Tensor(const vector<Variable> &_shape, TensorType _type, DataType _dataType, void* _ptr, string name):
        tensorType(_type), index(-1), dataType(_dataType), data_ptr(_ptr), sub_op_type(DATA), shape(_shape){
    if(name.empty()){
        this->name = std::move(Producer::get_unique_tensor_name());
    }else{
        this->name = std::move(name);
    }
}

void Tensor::print_data(ostream& out) {
    out<<to_string()<<endl;
    int size=1;
    for(auto & i : shape){
        size*=i.get_upper_bound();
    }
    assert(data_ptr);
    for(int i=0;i<size;i++){
        if(dataType==DT_FLOAT) out<<*((float*)data_ptr+i)<<" ";
    }
    out<<endl;
}

SubOpMetaType Tensor::convert_sub_op_type(SubOpType _op) {
    switch (_op) {
        case ADD:
        case MUL:
        case NEG:
            return COMPUTATION;
        case ASSIGN:
        case COND:
            return MEMORY;
        case SUM:
        case MAX:
            return REDUCE;
        case SMAX:
        case LOG:
        case EXP:
        case REC:
        case SQRT:
            //todo refactor: construct Nonlinear class, eliminate log exp sqrt rec
            return NONLINEAR;
        case TRANSFORM:
            return LINEAR;
        case DATA:
            return ELEMENT;
        case NOP:
        case RESHAPE:
            return PSEUDO;
        default:
            throw exception();
    }
}

SupportedDataType Tensor::convert_data_type(DataType t) {
    switch (t) {
        case DT_FLOAT:
        case DT_DOUBLE:
        case DT_HALF:
            return SupportedDataType::SDT_FLOAT;
        case DT_INT8:
        case DT_INT16:
        case DT_INT32:
        case DT_INT64:
            return SupportedDataType::SDT_INT;
        case DT_UINT8:
        case DT_UINT16:
        case DT_UINT32:
        case DT_UINT64:
            return SupportedDataType::SDT_UINT;
        default:
            return SDT_INVALID;
    }
}

bool Tensor::is_same_initial_weight(const Tensor &t) const {
    if(sub_op_type!=DATA||tensorType!=WEIGHT) return false;
    if(this->shape!=t.shape) return false;
    int size = get_length();
    assert(data_ptr&&t.data_ptr);
    for(int i=0;i<size;i++){
        //convert bits to float to judge if the two is equal
        if(abs(*(float *)data_ptr - *(float*)t.data_ptr)>1e-3) return false;
    }
    return true;
}

int Tensor::get_length() const{
    int ans=1;
    for(auto&b:shape){
        ans*=b.get_upper_bound();
    }
    return ans;
}

void Tensor::reshape(const vector<Variable>& new_shape) {
    //check
    int size=1,size1=1;
    for(auto& v:shape){
        size*=v.get_upper_bound();
    }
    for(auto&v:new_shape){
        size1*=v.get_upper_bound();
    }
    if(size!=1&&size1!=1) assert(size==size1);
    original_shape = this->shape;
    this->shape = new_shape;
}

void Tensor::reshape(const vector<int>& new_shape) {
    int size = 1;
    for(const auto& v:this->shape){
        size *=v.get_upper_bound();
    }
    //Assuming only one -1 at new_shape ,not checking yet
    int new_size=1;
    int neg_index = -1;
    for(int i=0;i<new_shape.size();i++){
        if(new_shape[i]==-1) neg_index = i;
        else new_size*=new_shape[i];
    }
    if(neg_index!=-1) const_cast<int&>(new_shape[neg_index]) = size/new_size; //assuming size%new_size==0
    vector<Variable> vars;
    if(this->tensorType==INPUT||this->tensorType==MIX){
        vars = Graph::get_variable(new_shape, true);
    }else{
        vars = Graph::get_variable(new_shape,false);
    }
    // check at Tensor
    this->reshape(vars);
}


Sum::Sum(TensorHandle  _out, TensorHandle  _in,const vector<Variable>& _v,int idx,const vector<int>& stride,bool bound)
        : Computation(_out,SUM,_in,idx), reduce_axis(_v), stride(stride){
    is_boundary = bound;
    shape_inference();
}
Max::Max(TensorHandle  _out, TensorHandle  _in, const vector<Variable>& _v,int idx):Computation(_out,MAX,_in,idx) {
    this->reduce_axis = _v;
    shape_inference();
}

string Sum::op_to_string() {
    string result;
    result+= sub_op_type_string[SUM] + "<";
    assert(!reduce_axis.empty());
    result += reduce_axis[0].name;
    for(int i=1;i<reduce_axis.size();i++)
        result+=","+reduce_axis[i].name;
    result+=">";
    return result;
}
string Max::op_to_string() {
    string result;
    result+= sub_op_type_string[this->op] + "<";
    assert(!reduce_axis.empty());
    result += reduce_axis[0].name;
    for(int i=1;i<reduce_axis.size();i++)
        result+=","+reduce_axis[i].name;
    result+=">";
    return result;
}

void Sum::shape_inference() {
    vector<Variable> v;
    for(auto & i : in[0]->shape){
        bool flag= true;
        for(auto & reduce_axi : reduce_axis){
            if(i==reduce_axi){
                flag = false;
                break;
            }
        }
        if(flag) v.push_back(i);
    }
    out->shape = std::move(v);
}

string Sum::to_compute() {
    string  result = this->out->name+" = te.compute(";
    result+= out->to_string(true,false,false,false,true);
    result+=", lambda ";
    for(int i=0;i<out->shape.size();i++){
        if(out->shape[i]==subst.first) result+= subst.second.name;
        else result+= out->shape[i].name;
        if(i!=out->shape.size()-1) result+=",";
    }
    if(!in[0]->serialization.empty()){
        result+=": te.sum("+in[0]->serialization+",axis = [";;
    }else{
        result+=": te.sum("+in[0]->to_string()+",axis = [";
    }
    for(int i=0;i<reduce_axis.size();i++){
        result+=reduce_axis[i].name;
        if(i!=reduce_axis.size()-1) result+=",";
    }
    result +="]), name='"+this->out->name+"',";
    if(!in[0]->attribute.empty()){
        result+=" attrs={";
        for(auto &p:in[0]->attribute) result+=p;
        result+="}";
    }
    result+=")";
    return result;
}

vector<string> Sum::to_reduce_axis() {
     vector<string> result;
    for(auto& p:reduce_axis){
        string tmp = p.name+" = te.reduce_axis(("+to_string(p.get_lower_bound())+","+to_string(p.get_upper_bound())+"),name='"+p.name+"')";
        result.push_back(tmp);
    }
    return result;
}
vector<string> Max::to_reduce_axis() {
    vector<string> result;
    for(auto& p:reduce_axis){
        string tmp = p.name+" = te.reduce_axis(("+to_string(p.get_lower_bound())+","+to_string(p.get_upper_bound())+"),name='"+p.name+"')";
        result.push_back(tmp);
    }
    return result;
}

string Sum::to_c_statement() {
    string result = out->name+"[";
    for(auto iter=out->shape.begin();iter!=out->shape.end();iter++){
        if(*iter==subst.first) result+=subst.second.name;
        else result+=iter->name;
        if(iter!=out->shape.end()-1) result+="][";
    }
    result+="]";
    result += "+="+in[0]->to_string(false,true)+";";
    return result;
}

string Sum::to_c_statement_initial() {
    string result = out->name+"[";
    for(auto iter=out->shape.begin();iter!=out->shape.end();iter++){
        if(*iter==subst.first) result+=subst.second.name;
        else result+=iter->name;
        if(iter!=out->shape.end()-1) result+="][";
    }
    result+="]";
    return result+" = 0;";
}
string Max::to_c_statement_initial() {
    string result = out->to_string(false,true,true,true,true)+"= - 1.0/0.0";
    return result;
}

vector<Variable> Sum::get_original_shape() {
     vector<Variable> result(this->out->shape.begin(),this->out->shape.end());
     for(auto&p:result){
         if(p==subst.first){p = subst.second;break;}
     }
     return result;
}

/*
 *   main compute allocate 0 and +=
 */
void Sum::compute() {

    SupportedDataType t = Tensor::convert_data_type(in[0]->dataType);
    this->out->data_ptr = allocate(t, this->out->shape); //
    int size = 1;
    for(auto& v:this->in[0]->shape){
        size*=v.get_upper_bound();
    }
    for(int i=0;i<size;i++){ // i is in_pos
        vector<int> in_index = Math::get_concrete_index(this->in[0]->shape, i);
        vector<int> index = Math::get_concrete_index(this->in[0]->shape, this->out->shape, in_index);

        int out_pos = Math::get_concrete_index(this->out->shape, index);
        switch (t) {
            case SDT_FLOAT:{
                auto tmp = (float *)((float *)out->data_ptr+out_pos);*tmp += *((float *)in[0]->data_ptr+i); break;
            }
            case SDT_INT:{
                auto tmp = (int *)((int *)out->data_ptr+out_pos);*tmp += *((int *)in[0]->data_ptr+i);break;
            }
            case SDT_UINT:{
                auto tmp = (uint *)((uint *)out->data_ptr+out_pos);*tmp += *((uint *)in[0]->data_ptr+i);break;
            }
            default: assert(false);
        }

    }
}

bool Sum::is_stride_equals(int value) {
    for(auto& i:stride){
        if(i!=value) return false;
    }
    return true;
}

void Sum::set_compute() {
    string result;
    if(!in[0]->serialization.empty()){
        result+="te.sum("+in[0]->serialization+",axis = [";;
    }else{
        result+="te.sum("+in[0]->to_string()+",axis = [";
    }
    for(int i=0;i<reduce_axis.size();i++){
        result+=reduce_axis[i].name;
        if(i!=reduce_axis.size()-1) result+=",";
    }
    result +="])";
    out->serialization = result;
    //pass in[0]'s attribute to out's attribute
    out->attribute = in[0]->attribute;
}

void Max::shape_inference() {
    vector<Variable> v;
    for(auto & i : in[0]->shape){
        bool flag= true;
        for(auto & reduce_axi : reduce_axis){
            if(i==reduce_axi){
                flag = false;
                break;
            }
        }
        if(flag) v.push_back(i);
    }
    out->shape = std::move(v);
}
/*
 *  main compute allocate -infinite and max
 */
void Max::compute() {
    SupportedDataType t = Tensor::convert_data_type(in[0]->dataType);
    this->out->data_ptr = allocate(t, this->out->shape,-infinite); //
    int size = 1;
    for(auto& v:this->in[0]->shape){
        size*=v.get_upper_bound();
    }
    for(int i=0;i<size;i++){ // i is in_pos
        vector<int> in_index = Math::get_concrete_index(this->in[0]->shape, i);
        vector<int> index = Math::get_concrete_index(this->in[0]->shape, this->out->shape, in_index);

        int out_pos = Math::get_concrete_index(this->in[0]->shape, index);
        switch (t) {
            case SDT_FLOAT:{
                auto tmp = (float *)((float *)out->data_ptr+out_pos);*tmp =  max(*tmp,*((float *)in[0]->data_ptr+i)) ; break;
            }
            case SDT_INT:{
                auto tmp = (int *)((int *)out->data_ptr+out_pos); *tmp =max(*tmp,*((int *)in[0]->data_ptr+i)) ; break;
            }
            case SDT_UINT:{
                auto tmp = (uint *)((uint *)out->data_ptr+out_pos);*tmp =  max(*tmp,*((uint *)in[0]->data_ptr+i)) ;break;
            }
            default: assert(false);
        }
    }
}

string Max::to_compute() {
    string  result = this->out->name+" = te.compute(";

    result+= out->to_string(true,false,false,false,true);
    result+=", lambda ";
    result+= out->to_string(false,false,false,true,false, true);
    if(!in[0]->serialization.empty()){
        result+=": te.max("+in[0]->serialization+",axis = [";;
    }else{
        result+=": te.max("+in[0]->to_string()+",axis = [";
    }
    for(int i=0;i<reduce_axis.size();i++){
        result+=reduce_axis[i].name;
        if(i!=reduce_axis.size()-1) result+=",";
    }
    result +="]), name='"+this->out->name+"',";
    if(!in[0]->attribute.empty()){
        result+=" attrs={";
        for(auto &p:in[0]->attribute) result+=p;
        result+="}";
    }
    result+=")";
    return result;
}

string Max::to_c_statement() {
    string result = out->to_string(false, true);

    result += "max("+out->to_string(false, true)+","+in[0]->to_string(false,true)+");";
    return result;
}

void Max::set_compute() {
    string result;
    if(!in[0]->serialization.empty()){
        result+="te.max("+in[0]->serialization+",axis = [";;
    }else{
        result+="te.max("+in[0]->to_string()+",axis = [";
    }
    for(int i=0;i<reduce_axis.size();i++){
        result+=reduce_axis[i].name;
        if(i!=reduce_axis.size()-1) result+=",";
    }
    result +="])";
    out->serialization = result;
    //pass in[0]'s attribute to out's attribute
    out->attribute = in[0]->attribute;
}

Mul::Mul(TensorHandle  _out, TensorHandle  _in0, TensorHandle  _in1,int idx,const vector<Variable>& simplify)
    :Computation(_out,MUL,_in0,_in1,idx),simplify_const_tensor_indices(simplify){
    // special case :mul 0D zero to accelerate preprocess_weights
    if(in[0]==&Tensor::zero||in[1]==&Tensor::zero) {
        out->shape.clear();
    }else{
        Computation::binary_shape_inference();
    }

}

void Mul::compute() {
    assert(in[0]->dataType==in[1]->dataType);
    SupportedDataType t = Tensor::convert_data_type(in[0]->dataType);
    this->out->data_ptr = allocate_random(t, this->out->shape); //
    int size = 1;
    for(auto& v:this->out->shape){
        size*=v.get_upper_bound();
    }
    //special case mul 1 :share memory rather than compute
    if(in[0]==&Tensor::one) {out->data_ptr = in[1]->data_ptr; return;}
    if(in[1]==&Tensor::one) {out->data_ptr = in[0]->data_ptr;return;}
    for(int i=0;i<size;i++){
        vector<int> index = Math::get_concrete_index(this->out->shape, i);
        vector<int> in0_index = Math::get_concrete_index(this->out->shape, this->in[0]->shape, index);
        vector<int> in1_index = Math::get_concrete_index(this->out->shape, this->in[1]->shape, index);
        int in0_pos = Math::get_concrete_index(this->in[0]->shape, in0_index);
        int in1_pos = Math::get_concrete_index(this->in[1]->shape, in1_index);
        switch (t) {
            case SDT_FLOAT:{
                auto tmp= (float *)out->data_ptr+i; *tmp = (*((float *)in[0]->data_ptr+in0_pos)) * (*((float *)in[1]->data_ptr+in1_pos));
                break;
            }
            case SDT_INT:{
                auto tmp= (int*)out->data_ptr+i; *tmp = (*((int*)in[0]->data_ptr+in0_pos)) * (*((int*)in[1]->data_ptr+in1_pos));
                break;
            }
            case SDT_UINT:{
                auto tmp= (uint*)out->data_ptr+i; *tmp = (*((uint*)in[0]->data_ptr+in0_pos)) * (*((uint*)in[1]->data_ptr+in1_pos));
                break;
            }
            default: assert(false);
        }
    }

}

string Mul::to_compute() {
    string result = this->out->name+" = te.compute(";

    result+= out->to_string(true,false,false,false,true);
    result+=", lambda ";
    result+= out->to_string(false,false,false,true,false);
    if(!in[0]->serialization.empty()) result+=": "+in[0]->serialization;
    else result+=": "+in[0]->to_string();
    if(!in[1]->serialization.empty()) result+=" * "+in[1]->serialization;
    else result+= " * "+in[1]->to_string();
    result +=", name='"+this->out->name+"',";
    if(simplify_const_tensor_indices.empty()){
        result+=")";
    }else{
        result+="attrs={'auto_scheduler_simplify_const_tensor_indices':[";
        for(int i=0;i<simplify_const_tensor_indices.size();i++){
            result+="'"+simplify_const_tensor_indices[i].name+"'";
            if(i!=simplify_const_tensor_indices.size()-1) result+=",";
        }
        result+="]},)";
    }
    return result;
}

string Mul::to_c_statement() {
    string result = out->to_string(false,true)+" = "+in[0]->to_string(false,true)+"*"+in[1]->to_string(false,true)+";";
    return result;
}

string Mul::op_to_string() {
    string result;
    result+= sub_op_type_string[this->op];
    if(!simplify_const_tensor_indices.empty()){
        result+="_simplify<";
        result += simplify_const_tensor_indices[0].name;
        for(int i=1;i<simplify_const_tensor_indices.size();i++)
            result+=","+simplify_const_tensor_indices[i].name;
        result+=">";
    }
    return result;
}

void Mul::set_compute() {
    string result;
    if(!in[0]->serialization.empty()) result+=in[0]->serialization;
    else result+=in[0]->to_string();
    if(!in[1]->serialization.empty()) result+=" * "+in[1]->serialization;
    else result+= " * "+in[1]->to_string();
    out->serialization = result;
    out->attribute = in[0]->attribute;
    if(!in[1]->attribute.empty())
        out->attribute.insert(out->attribute.end(),in[1]->attribute.begin(),in[1]->attribute.end());
    if(!simplify_const_tensor_indices.empty()){
        string simplify = "'auto_scheduler_simplify_const_tensor_indices':[";
        for(int i=0;i<simplify_const_tensor_indices.size();i++){
            simplify+="'"+simplify_const_tensor_indices[i].name+"'";
            if(i!=simplify_const_tensor_indices.size()-1) simplify+=",";
        }
        simplify+="]";
        out->attribute.insert(out->attribute.end(),simplify);
    }
}

string Assign::op_to_string() {
    string result = sub_op_type_string[op] + "<";
    for(int i=0;i<this->lambda.size();i++){
        result+=this->lambda[i].to_string();
        if(i!=this->lambda.size()-1) result+=",";
    }
    result+=">";
    return result;
}
Assign::Assign(TensorHandle  _out,TensorHandle  _in,const vector<Variable>& out_shape,int idx,const vector<Affine>& in_lambda)
        :Computation(_out,ASSIGN,_in,idx),lambda(in_lambda){
    _out->shape = out_shape;
    assert(in_lambda.size()==_in->shape.size());
    int i=0;
    vector<Constraint> v;
    for(auto&a:this->lambda){
        if(a.type==AffineType::AFFINE_VARIABLE&&in[0]->shape[i].name!=a.to_string()) mapping[_in->shape[i]] = a;
        if(a.type==AffineType::AFFINE_EXPR){
            mapping[_in->shape[i]] = a;
            v.emplace_back(Constraint(CONSTRAINT_GE,Affine(a),Affine(_in->shape[i].get_lower_bound())));
            v.emplace_back(Constraint(CONSTRAINT_LT,Affine(a),Affine(_in->shape[i].get_upper_bound())));
        }
        i++;
    }
    formula = Formula::create_all(v);
}

Assign::Assign(TensorHandle  _out, TensorHandle  _in, const vector<Variable>& out_shape, int idx
               ,const vector<Affine>& in_lambda,Formula* ptr):Computation(_out,ASSIGN,_in,idx),lambda(in_lambda) {
    _out->shape = out_shape;
    assert(in_lambda.size()==_in->shape.size());
    int i=0;
    for(auto&a:this->lambda) {
        if (a.type == AffineType::AFFINE_EXPR||
                (a.type == AffineType::AFFINE_VARIABLE && in[0]->shape[i].name != a.to_string())){
            mapping[_in->shape[i]] = a;
        }
        i++;
    }
    //user need to specify all formula including bound and self-defined formula
    formula = ptr;
    if(ptr) formula->autogen = false;
}
Assign::Assign(TensorHandle _out,TensorHandle _in,const vector<Variable>& out_shape,int idx,const map<Variable,Affine>& _mapping)
    :Computation(_out,ASSIGN,_in,idx),mapping(_mapping){
    assert(!_mapping.empty());
    _out->shape = out_shape;
    vector<Constraint> constraints;
    for(auto&v:_in->shape){
        if(mapping.find(v)==mapping.end()){
            lambda.emplace_back(v);
        }else{
            Affine& temp = this->mapping[v];
            lambda.emplace_back(temp);
            constraints.emplace_back(Constraint(CONSTRAINT_GE,Affine(temp),Affine(v.get_lower_bound())));
            constraints.emplace_back(Constraint(CONSTRAINT_LT,Affine(temp),Affine(v.get_upper_bound())));
        }
    }
    formula = Formula::create_all(constraints);
}
Assign::Assign(TensorHandle _out,TensorHandle _in,const vector<Variable>& out_shape,int idx,const map<Variable,Affine>& _mapping,Formula* ptr)
        :Computation(_out,ASSIGN,_in,idx),mapping(_mapping){//make sure _mapping not changing
    assert(!_mapping.empty());
    _out->shape = out_shape;
    for(auto&v:_in->shape){
        if(mapping.find(v)==mapping.end()){
            lambda.emplace_back(v);
        }else{
            Affine& temp = this->mapping[v];
            lambda.emplace_back(temp);
        }
    }
    // Note that now if ptr is not nullptr ,user need to specify all formula including bound and self-defined formula
    formula = ptr;
    if(ptr) formula->autogen=false;
}

//Assign::Assign(TensorHandle  _out, TensorHandle  _in, const vector<Variable>& out_shape, const vector<Variable>& pad_axis, const vector<int>& pads, int idx,
//               const vector<Variable>& old):Computation(_out,ASSIGN,_in,idx) {
//    assert(pad_axis.size()==pads.size());
//    vector<Affine> v;
//    for(auto & i : out_shape){
//        bool found=false;
//        int j=0;
//        for(;j<pad_axis.size();j++){
//            if(i==pad_axis[j]){found = true;break;}
//        }
//        if(found){
//            map<string,int> eval;
//            string expr = pad_axis[j].name+"-"+::to_string(pads[j]);
//            v.push_back(*Parser::make_affine(expr,eval));
//        }else{
//            v.emplace_back(i);
//        }
//    }
//    _out->setShape(out_shape);
//    is_pad = true;
//    this->pads = pads;
//    this->old_axis = old;
//    this->pad_axis = pad_axis;
//    this->lambda = v;
//    init_condition();
//}

void Assign::compute() {
    SupportedDataType t = Tensor::convert_data_type(in[0]->dataType);
    this->out->data_ptr = allocate(t, this->out->shape,0);
    int size = 1;
    for(auto& v:this->out->shape){
        size*=v.get_upper_bound();
    }
    for(int i=0;i<size;i++){
        vector<int> index = Math::get_concrete_index(this->out->shape, i); //index is out_shape  index
        vector<int> in_index = Math::get_concrete_index(this->lambda, this->out->shape, index); // lambda size == in size
        map<string,int> eval_map;
        for(int j=0;j<this->out->shape.size();j++){
            eval_map[out->shape[j].name] = index[j];
        }
        if(this->formula->evaluate(eval_map)){
            int in_pos = Math::get_concrete_index(this->in[0]->shape, in_index);
            switch (t) {
                case SDT_FLOAT:{
                    auto tmp = (float *)((float *)out->data_ptr+i);*tmp = *((float *)in[0]->data_ptr+in_pos); break;
                }
                case SDT_INT:{
                    auto tmp = (int *)((int *)out->data_ptr+i);*tmp = *((int *)in[0]->data_ptr+in_pos);break;
                }
                case SDT_UINT:{
                    auto tmp = (uint *)((uint *)out->data_ptr+i);*tmp = *((uint *)in[0]->data_ptr+in_pos);break;
                }
                default: assert(false);
            }
        }
    }
}

string Assign::to_compute() {
    string target ;
    if(!in[0]->serialization.empty()) target = in[0]->serialization;
    else{
        target =  in[0]->name +"[";
        for(int i=0;i<lambda.size();i++){
            target+= lambda[i].to_string(true);
            if(i != lambda.size() - 1) target+=",";
        }
        target +="]";
    }

    string result = this->out->name+" = te.compute(";
    result+= out->to_string(true,false,false,false,true);
    result+=", lambda ";
    result+= out->to_string(false,false,false,true,false, true)+": ";
    if(formula) result+="te.if_then_else("+formula->to_string(true)+","+target+",0)";
    else result+=target;
    result+=",name = '"+this->out->name+"')";
    return result;
}

string Assign::to_c_statement() {
    string branch = out->to_string(false,true)+" = "+in[0]->name+"[";
    for(int i=0;i<lambda.size();i++){
        branch+= lambda[i].to_string();
        if(i != lambda.size() - 1) branch+="][";
    }
    branch+= "];";
    string result;
    string cond;
    if(formula){
        cond= formula->to_string_c_style();
        result = "if("+cond+") "+branch+" else "+out->to_string(false,true)+" = 0 ;";
    }else{
        result = branch;
    }
    return result;
}

bool Assign::is_stride_equals(int value) {
    for(auto&a:lambda){
        if(!a.is_coefficients_equals(value)) return false;
    }
    return true;
}

void Assign::set_compute() {
    string target ;
    if(!in[0]->serialization.empty()) target = in[0]->serialization;
    else{
        target =  in[0]->name +"[";
        for(int i=0;i<lambda.size();i++){
            target+= lambda[i].to_string(true);
            if(i != lambda.size() - 1) target+=",";
        }
        target +="]";
    }
    string result;
    if(formula) result+="te.if_then_else("+formula->to_string(true)+","+target+",0)";
    else result+=target;
    out->serialization = result;
}

Add::Add(TensorHandle   _out, TensorHandle  _in0, TensorHandle  _in1, int idx):Computation(_out,ADD,_in0,_in1,idx) {
    Computation::binary_shape_inference();
}

SMax::SMax(TensorHandle  _out, TensorHandle  _in0, TensorHandle  _in1, int idx) :Computation(_out,SMAX,_in0,_in1,idx) {
    Computation::binary_shape_inference();
//    if(_in1->shape.empty()) _out->tensorType = _in0->tensorType; // feeling not need this
}

void Add::compute() {
//    assert(in[0]->dataType==in[1]->dataType);
    SupportedDataType t = Tensor::convert_data_type(in[0]->dataType);
    this->out->data_ptr = allocate_random(t, this->out->shape); //
    int size = 1;
    for(auto& v:this->out->shape){
        size*=v.get_upper_bound();
    }
    //special case add by 0 :share memory rather than compute
    if(in[0]==&Tensor::zero) {out->data_ptr = in[1]->data_ptr;return;}
    if(in[1]==&Tensor::zero){out->data_ptr = in[0]->data_ptr;return;}
    for(int i=0;i<size;i++){
        vector<int> index = Math::get_concrete_index(this->out->shape, i);
        vector<int> in0_index = Math::get_concrete_index(this->out->shape, this->in[0]->shape, index);
        vector<int> in1_index = Math::get_concrete_index(this->out->shape, this->in[1]->shape, index);
        int in0_pos = Math::get_concrete_index(this->in[0]->shape, in0_index);
        int in1_pos = Math::get_concrete_index(this->in[1]->shape, in1_index);
        switch (t) {
            case SDT_FLOAT:{
                auto tmp= (float *)out->data_ptr+i; *tmp = (*((float *)in[0]->data_ptr+in0_pos)) + (*((float *)in[1]->data_ptr+in1_pos));
                break;
            }
            case SDT_INT:{
                auto tmp= (int*)out->data_ptr+i; *tmp = (*((int*)in[0]->data_ptr+in0_pos)) + (*((int*)in[1]->data_ptr+in1_pos));
                break;
            }
            case SDT_UINT:{
                auto tmp= (uint*)out->data_ptr+i; *tmp = (*((uint*)in[0]->data_ptr+in0_pos)) + (*((uint*)in[1]->data_ptr+in1_pos));
                break;
            }
            default: assert(false);
        }
    }
}

string Add::to_compute() {
    string result = this->out->name+" = te.compute(";

    result+= out->to_string(true,false,false,false,true);
    result+=", lambda ";
    result+= out->to_string(false,false,false,true,false, true);
    if(!in[0]->serialization.empty()) result+=": "+in[0]->serialization;
    else result+=": "+in[0]->to_string();
    if(!in[1]->serialization.empty()) result+=" + "+in[1]->serialization;
    else result+= " + "+in[1]->to_string();
    result +=", name='"+this->out->name+"')";
    return result;
}

string Add::to_c_statement() {
    string result = out->to_string(false,true)+" = "+in[0]->to_string(false,true)+"+"+in[1]->to_string(false,true)+";";
    return result;
}

void Add::set_compute() {
    string result;
    if(!in[0]->serialization.empty()) result+=in[0]->serialization;
    else result+=in[0]->to_string();
    if(!in[1]->serialization.empty()) result+=" * "+in[1]->serialization;
    else result+= " * "+in[1]->to_string();
    out->serialization = result;
    out->attribute = vector<string>(in[0]->attribute);
    out->attribute.insert(out->attribute.end(),in[1]->attribute.begin(),in[1]->attribute.end());
}

void SMax::compute() {
    assert(in[0]->dataType==in[1]->dataType);
    SupportedDataType t = Tensor::convert_data_type(in[0]->dataType);
    this->out->data_ptr = allocate_random(t, this->out->shape); //
    int size = 1;
    for(auto& v:this->out->shape){
        size*=v.get_upper_bound();
    }
    for(int i=0;i<size;i++){
        vector<int> index = Math::get_concrete_index(this->out->shape, i);
        vector<int> in0_index = Math::get_concrete_index(this->out->shape, this->in[0]->shape, index);
        vector<int> in1_index = Math::get_concrete_index(this->out->shape, this->in[1]->shape, index);
        int in0_pos = Math::get_concrete_index(this->in[0]->shape, in0_index);
        int in1_pos = Math::get_concrete_index(this->in[1]->shape, in1_index);
        switch (t) {
            case SDT_FLOAT:{
                auto tmp= (float *)out->data_ptr+i; *tmp = max((*((float *)in[0]->data_ptr+in0_pos)) , (*((float *)in[1]->data_ptr+in1_pos)));
                break;
            }
            case SDT_INT:{
                auto tmp= (int*)out->data_ptr+i; *tmp = max((*((int*)in[0]->data_ptr+in0_pos)) , (*((int*)in[1]->data_ptr+in1_pos)));
                break;
            }
            case SDT_UINT:{
                auto tmp= (uint*)out->data_ptr+i; *tmp = max((*((uint*)in[0]->data_ptr+in0_pos)) , (*((uint*)in[1]->data_ptr+in1_pos)));
                break;
            }
            default: assert(false);
        }
    }
}

string SMax::to_compute() {
    bool x_has_bracket = true,y_has_bracket=true;
    if(in[0]->shape.empty())  x_has_bracket = false;
    if(in[1]->shape.empty()) y_has_bracket = false;
    string x,y;
    if(!in[0]->serialization.empty()) x = in[0]->serialization;
    else x=in[0]->to_string(false,false,true,true,x_has_bracket);
    if(!in[1]->serialization.empty()) y = in[1]->serialization;
    else y=in[1]->to_string(false,false,true,true,y_has_bracket);
    //============<x>============<y>==============
    string result = this->out->name+" = te.compute(";
    result+= out->to_string(true,false,false,false,true);
    result+=", lambda ";
    result+= out->to_string(false,false,false,true,false, true);
    result+=": te.if_then_else("+x+">"+y+","+x+","+y+")"
             ", name='"+this->out->name+"')";
    return result;
}

string SMax::to_c_statement() {
    return out->to_string(false,true)+" = max("+in[0]->to_string(false,true)+","+in[1]->to_string(false,true)+");";
}

void SMax::set_compute() {
    bool x_has_bracket = true,y_has_bracket=true;
    if(in[0]->shape.empty())  x_has_bracket = false;
    if(in[1]->shape.empty()) y_has_bracket = false;
    string x,y;
    if(!in[0]->serialization.empty()) x = in[0]->serialization;
    else x=in[0]->to_string(false,false,true,true,x_has_bracket);
    if(!in[1]->serialization.empty()) y = in[1]->serialization;
    else y=in[1]->to_string(false,false,true,true,y_has_bracket);
    string result="te.if_then_else("+x+">"+y+","+x+","+y+")";
    out->serialization = result;
}

Nop::Nop(TensorHandle  _out, TensorHandle  _in, int idx):Computation(_out,NOP,_in,idx) {
    Computation::unary_shape_inference();
}

//Cond::Cond(TensorHandle _out , vector<Expr>& _exprs,TensorHandle _in0,TensorHandle _in1,const vector<Variable>& out_shape,int idx,
//           const vector<Expr>& in1_expr,const vector<Expr>& in2_expr):Computation(_out,COND,_in0,_in1,idx) {
//    this->cond_expr = _exprs;
//    this->out->setShape(out_shape);
//    this->in[0]->curExpr = _exprs[0];
//    this->in[1]->curExpr = _exprs[1];
//
//    if(in1_expr.empty()){
//        this->in[0]->lambda = _in0->lambda;
//    }else{
//        _in0->lambda = in1_expr;
//        //this->in[0]->lambda = in1_expr;
//    }
//    int i=0;
//    for(auto &item:this->in[0]->lambda){
//        if(!item.isVariable||!item.is_equal_to_variable(_in0->shape[i])) {
//            this->in[0]->curVariable = _in0->shape[i];
//            break;
//        }
//        i++;
//    }
//    if(in2_expr.empty()){
//        this->in[1]->lambda = _in1->lambda;
//    }else{
//        //this->in[1]->lambda = in2_expr;
//        _in1->lambda = in2_expr;
//    }
//    i=0;
//    for(auto &item:this->in[1]->lambda){
//        if(!item.isVariable||!item.is_equal_to_variable(_in1->shape[i])) {
//            this->in[1]->curVariable = _in1->shape[i];
//            break;
//        }
//        i++;
//    }
//
//}

string Cond::op_to_string() {
    string result = sub_op_type_string[op] + "<";
    result+= this->in0_axis.name+",";
    result+=this->in1_axis.name+">";
    return result;
}

Cond::Cond(TensorHandle _out, TensorHandle _in0, TensorHandle _in1, const Variable& v1,const Variable & v2 ,const Variable& vo,int idx)
        :Computation(_out,COND,_in0,_in1,idx),in0_axis(v1),in1_axis(v2),new_axis(vo) {
    // cond legal check
    assert(in[0]->shape.size()==in[1]->shape.size());
    vector<Variable> out_shape;
    for(int i=0;i<in[0]->shape.size();i++){
        if(in[0]->shape[i]!=in0_axis){
            assert(in[0]->shape[i]==in[1]->shape[i]);
            out_shape.push_back(in[0]->shape[i]);
        }else{
            assert(in[1]->shape[i]==in1_axis);
            out_shape.push_back(new_axis);
        }
    }
    out->shape = std::move(out_shape);
}

void Cond::compute() {
    assert(in[0]->dataType==in[1]->dataType);
    SupportedDataType t = Tensor::convert_data_type(in[0]->dataType);
    this->out->data_ptr = allocate_random(t, this->out->shape); //
    int size = 1;
    for(auto& v:this->out->shape){
        size*=v.get_upper_bound();
    }
    // in[0] in[1] out ,shape size equals each other
    int pos=0;
    for(int i=0;i<this->out->shape.size();i++){
        if(out->shape[i]==new_axis) {pos = i;break;}
    }
    for(int i=0;i<size;i++){
        vector<int> index = Math::get_concrete_index(this->out->shape, i);
        if(index[pos]<in0_axis.get_upper_bound()){
            int in0_index = Math::get_concrete_index(this->in[0]->shape, index);
            switch (t) {
                case SDT_FLOAT:{
                    auto tmp = (float *)((float *)out->data_ptr+i); *tmp = *((float *)in[0]->data_ptr+in0_index);
                    break;
                }
                case SDT_INT:{
                    auto tmp = (int*)((int*)out->data_ptr+i);   *tmp = *((int*)in[0]->data_ptr+in0_index);
                    break;
                }
                case SDT_UINT:{
                    auto tmp = (uint*)((uint*)out->data_ptr+i);   *tmp = *((uint *)in[0]->data_ptr+in0_index);
                    break;
                }
                default: assert(false);
            }
        }else{
            index[pos] -= in0_axis.get_upper_bound();
            int in1_index = Math::get_concrete_index(this->in[1]->shape, index);

            switch (t) {
                case SDT_FLOAT:{
                    auto tmp = (float *)((float *)out->data_ptr+i);*tmp = *((float *)in[1]->data_ptr+in1_index);
                    break;
                }
                case SDT_INT:{
                    auto tmp = (int*)((int*)out->data_ptr+i);*tmp = *((int*)in[1]->data_ptr+in1_index);
                    break;
                }
                case SDT_UINT:{
                    auto tmp = (uint*)((uint*)out->data_ptr+i);*tmp = *((uint*)in[1]->data_ptr+in1_index);
                    break;
                }
                default: assert(false);
            }
        }
    }
}

string Cond::to_compute() {
    string x,y;
    if(!in[0]->serialization.empty()) x = in[0]->serialization;
    else{
        x=in[0]->name+"[";
        for(int i=0; i < in[0]->shape.size(); i++){
            if(in[0]->shape[i]==in0_axis) x+= new_axis.name;
            else x+=in[0]->shape[i].name;
            if(i != in[0]->shape.size() - 1) x+=",";
        }
        x+= "]";
    }
    if(!in[1]->serialization.empty()) y = in[1]->serialization;
    else{
        y=in[1]->name+"[";
        for(int i=0; i < in[1]->shape.size(); i++){
            if(in[1]->shape[i]==in1_axis) y+= new_axis.name+"-"+::to_string(in0_axis.get_upper_bound());
            else y+=in[1]->shape[i].name;
            if(i != in[1]->shape.size() - 1) y+=",";
        }
        y +="]";
    }
    //============<x>============<y>==============
    string result = this->out->name+" = te.compute(";

    result +="(";
    for(int i=0; i < out->shape.size(); i++){
        result+=::to_string(out->shape[i].get_upper_bound());
        if(i != out->shape.size() - 1) result+=",";
    }
    result+="), lambda ";
    for(int i=0; i < out->shape.size(); i++){
        result+=out->shape[i].name;
        if(i != out->shape.size() - 1) result+=",";
    }
    result+=": te.if_then_else("+new_axis.name+"<"+::to_string(in0_axis.get_upper_bound())+","+x+","+y+")"
                                                     ", name='"+this->out->name+"')";
    return result;
}

string Cond::to_c_statement() {
    string x=in[0]->name+"[";
    for(int i=0; i < in[0]->shape.size(); i++){
        if(in[0]->shape[i]==in0_axis) x+= new_axis.name;
        else x+=in[0]->shape[i].name;
        if(i != in[0]->shape.size() - 1) x+="][";
    }
    x+= "]";
    string y=in[1]->name+"[";
    for(int i=0; i < in[1]->shape.size(); i++){
        if(in[1]->shape[i]==in1_axis) y+= new_axis.name+"-"+::to_string(in0_axis.get_upper_bound());
        else y+=in[1]->shape[i].name;
        if(i != in[1]->shape.size() - 1) y+="][";
    }
    y +="]";
    string result;
    string cond;
    cond= new_axis.name+"<"+::to_string(in0_axis.get_upper_bound());
    string branch1 = out->to_string(false,true)+" = "+x+";";
    string branch2 = out->to_string(false,true)+" = "+y+";";
    result = "if("+cond+") "+branch1+" else "+branch2;
    return result;
}

bool Cond::check_cond(const vector<Variable>& shape,const vector<Variable>& shape2) {
    if(shape.size()!=shape2.size())return false;
    int count=0;
    for(int i=0;i<shape.size();i++){
        if(shape[i]!=shape2[i]){
            count++;
        }
    }
    return count<=1;
}

bool Cond::check_cond(const vector<Variable>& shape,const vector<Variable>& shape2,const Variable& axis0,const Variable& axis1) {
    if(shape.size()!=shape2.size()) return false;
    for(int i=0;i<shape.size();i++){
        if(shape[i]==axis0){
            if(shape2[i]!=axis1) return false;
        }else if(shape2[i]!=shape[i]){
             return false;
        }
    }
    return true;
}

void Cond::set_compute() {
    string x,y;
    if(!in[0]->serialization.empty()) x = in[0]->serialization;
    else{
        x=in[0]->name+"[";
        for(int i=0; i < in[0]->shape.size(); i++){
            if(in[0]->shape[i]==in0_axis) x+= new_axis.name;
            else x+=in[0]->shape[i].name;
            if(i != in[0]->shape.size() - 1) x+=",";
        }
        x+= "]";
    }
    if(!in[1]->serialization.empty()) y = in[1]->serialization;
    else{
        y=in[1]->name+"[";
        for(int i=0; i < in[1]->shape.size(); i++){
            if(in[1]->shape[i]==in1_axis) y+= new_axis.name+"-"+::to_string(in0_axis.get_upper_bound());
            else y+=in[1]->shape[i].name;
            if(i != in[1]->shape.size() - 1) y+=",";
        }
        y +="]";
    }
    out->serialization = "te.if_then_else("+new_axis.name+"<"+::to_string(in0_axis.get_upper_bound())+","+x+","+y+")";
}

//void Cond::set_curVariable(const Variable& a,const Variable& b) {
//    this->in[0]->curVariable = a;
//    this->in[1]->curVariable = b;
//}

Neg::Neg(TensorHandle  _out,  TensorHandle  in, int idx):Computation(_out,NEG,in,idx) {
    Computation::unary_shape_inference();
}
Sqrt::Sqrt(TensorHandle  _out,  TensorHandle  in, int idx) :Computation(_out,SQRT,in,idx) {
    Computation::unary_shape_inference();
}

void Sqrt::compute() {
    SupportedDataType t = Tensor::convert_data_type(in[0]->dataType);
    this->out->data_ptr = allocate_random(t, this->out->shape); //
    int size = 1;
    for(auto& v:this->out->shape){
        size*=v.get_upper_bound();
    }
    for(int i=0;i<size;i++){
        switch (t) {
            case SDT_FLOAT:{
                *((float *)out->data_ptr+i) = sqrt(*((float *)in[0]->data_ptr+i));
                break;
            }
            case SDT_INT:{
                *((int*)out->data_ptr+i) = sqrt(*((int*)in[0]->data_ptr+i));
                break;
            }
            case SDT_UINT:{
                *((uint*)out->data_ptr+i) = sqrt(*((uint*)in[0]->data_ptr+i));
                break;
            }
            default: assert(false);
        }
    }
}

string Sqrt::to_compute() {
    string result = this->out->name+" = te.compute(";
    result+= out->to_string(true,false,false,false,true);
    result+=", lambda ";
    result+= out->to_string(false,false,false,true,false);
    if(!in[0]->serialization.empty()) result+=": te.sqrt("+in[0]->serialization+")";
    else result+=": te.sqrt("+in[0]->to_string()+")";
    result +=", name='"+this->out->name+"')";
    return result;
}

string Sqrt::to_c_statement() {
    return out->to_string(false,true)+" = sqrt("+in[0]->to_string(false,true)+");";
}

void Sqrt::set_compute() {
    string result;
    if(!in[0]->serialization.empty()) result+="te.sqrt("+in[0]->serialization+")";
    else result+="te.sqrt("+in[0]->to_string()+")";
    out->serialization = result;
}

void Neg::compute() {
    SupportedDataType t = Tensor::convert_data_type(in[0]->dataType);
    this->out->data_ptr = allocate_random(t, this->out->shape); //
    int size = 1;
    for(auto& v:this->out->shape){
        size*=v.get_upper_bound();
    }
    for(int i=0;i<size;i++){
        switch (t) {
            case SDT_FLOAT:{
                *((float *)out->data_ptr+i) = - *((float *)in[0]->data_ptr+i);
                break;
            }
            case SDT_INT:{
                *((int*)out->data_ptr+i) = - *((int*)in[0]->data_ptr+i);
                break;
            }
            case SDT_UINT:{
                *((uint*)out->data_ptr+i)= - *((uint*)in[0]->data_ptr+i);
                break;
            }
            default: assert(false);
        }
    }
}

string Neg::to_compute() {
    string result = this->out->name+" = te.compute(";
    result+= out->to_string(true,false,false,false,true);
    result+=", lambda ";
    result+= out->to_string(false,false,false,true,false);
    if(!in[0]->serialization.empty()) result+=": -"+in[0]->serialization;
    else result+=": -"+in[0]->to_string();
    result +=", name='"+this->out->name+"')";
    return result;
}

string Neg::to_c_statement() {
    return out->to_string(false,true)+" = -"+in[0]->to_string(false,true)+";";
}

void Neg::set_compute() {
    string result;
    if(!in[0]->serialization.empty()) result+="-"+in[0]->serialization;
    else result+="-"+in[0]->to_string();
    out->serialization = result;
}

Log::Log(TensorHandle  _out,  TensorHandle  in, int idx):Computation(_out,LOG,in,idx) {
    Computation::unary_shape_inference();
}

void Log::compute() {
    SupportedDataType t = Tensor::convert_data_type(in[0]->dataType);
    this->out->data_ptr = allocate_random(t, this->out->shape); //
    int size = 1;
    for(auto& v:this->out->shape){
        size*=v.get_upper_bound();
    }
    for(int i=0;i<size;i++){
        switch (t) {
            case SDT_FLOAT:{
                *((float *)out->data_ptr+i) = log(*((float *)in[0]->data_ptr+i));
                break;
            }
            case SDT_INT:{
                *((int*)out->data_ptr+i) = log(*((int*)in[0]->data_ptr+i));
                break;
            }
            case SDT_UINT:{
                *((uint*)out->data_ptr+i) = log(*((uint*)in[0]->data_ptr+i));
                break;
            }
            default: assert(false);
        }
    }
}

string Log::to_compute() {
    string result = this->out->name+" = te.compute(";
    result+= out->to_string(true,false,false,false,true);
    result+=", lambda ";
    result+= out->to_string(false,false,false,true,false);
    if(!in[0]->serialization.empty()) result+=": te.log("+in[0]->serialization+")";
    else result+=": te.log("+in[0]->to_string()+")";
    result +=", name='"+this->out->name+"')";
    return result;
}

string Log::to_c_statement() {
    return out->to_string(false,true)+" = log("+in[0]->to_string(false,true)+");";
}

void Log::set_compute() {
    string result;
    if(!in[0]->serialization.empty()) result+="te.log("+in[0]->serialization+")";
    else result+="te.log("+in[0]->to_string()+")";
    out->serialization = result;
}

Exp::Exp(TensorHandle  _out,  TensorHandle  in, int idx):Computation(_out,EXP,in,idx) {
    Computation::unary_shape_inference();
}

void Exp::compute() {
    SupportedDataType t = Tensor::convert_data_type(in[0]->dataType);
    this->out->data_ptr = allocate_random(t, this->out->shape); //
    int size = 1;
    for(auto& v:this->out->shape){
        size*=v.get_upper_bound();
    }
    for(int i=0;i<size;i++){
        switch (t) {
            case SDT_FLOAT:{
                *((float *)out->data_ptr+i) = exp(*((float *)in[0]->data_ptr+i));
                break;
            }
            case SDT_INT:{
                *((int*)out->data_ptr+i) = exp(*((int*)in[0]->data_ptr+i));
                break;
            }
            case SDT_UINT:{
                *((uint*)out->data_ptr+i) = exp(*((uint*)in[0]->data_ptr+i));
                break;
            }
            default: assert(false);
        }
    }
}

string Exp::to_compute() {
    string result = this->out->name+" = te.compute(";
    result+= out->to_string(true,false,false,false,true);
    result+=", lambda ";
    result+= out->to_string(false,false,false,true,false);
    if(!in[0]->serialization.empty()) result+=": te.exp("+in[0]->serialization+")";
    else result+=": te.exp("+in[0]->to_string()+")";
    result +=", name='"+this->out->name+"')";
    return result;
}

string Exp::to_c_statement() {
    return out->to_string(false,true)+" = exp("+in[0]->to_string(false,true)+");";
}

void Exp::set_compute() {
    string result;
    if(!in[0]->serialization.empty()) result+="te.exp("+in[0]->serialization+")";
    else result+="te.exp("+in[0]->to_string()+")";
    out->serialization = result;
}

Rec::Rec(TensorHandle  _out, TensorHandle  in, int idx):Computation(_out,REC,in,idx) {
    Computation::unary_shape_inference();
}

void Rec::compute() {
    SupportedDataType t = Tensor::convert_data_type(in[0]->dataType);
    this->out->data_ptr = allocate_random(t, this->out->shape); //
    int size = 1;
    for(auto& v:this->out->shape){
        size*=v.get_upper_bound();
    }
    for(int i=0;i<size;i++){
        switch (t) {
            case SDT_FLOAT:{
                *((float *)out->data_ptr+i) = float(1.0)/ *((float *)in[0]->data_ptr+i);
                break;
            }
            case SDT_INT:{
                *((int*)out->data_ptr+i) = 1/ *((int*)in[0]->data_ptr+i); // maybe resulting 0
                break;
            }
            case SDT_UINT:{
                *((uint*)out->data_ptr+i) = 1/ *((uint*)in[0]->data_ptr+i);//maybe resulting 0
                break;
            }
            default: assert(false);
        }
    }
}

string Rec::to_compute() {
    string result = this->out->name+" = te.compute(";
    result+= out->to_string(true,false,false,false,true);
    result+=", lambda ";
    result+= out->to_string(false,false,false,true,false);
    if(!in[0]->serialization.empty()) result+=": 1/"+in[0]->serialization;
    else result+=": 1/"+in[0]->to_string();
    result +=", name='"+this->out->name+"')";
    return result;
}

string Rec::to_c_statement() {
    return out->to_string(false,true)+" = 1.0/"+in[0]->to_string(false,true)+";";
}

void Rec::set_compute() {
    string result;
    if(!in[0]->serialization.empty()) result+="1.0/"+in[0]->serialization;
    else result+="1.0/"+in[0]->to_string();
    out->serialization = result;
}

//void Div::shape_inference() {
//    out->setShape(in[0]->shape);
//}
//
//Div::Div(TensorHandle  _out, TensorHandle  _in,float _d, int idx):Computation(_out,DIV,_in,idx) {
//    this->divider = _d;
//    //shape_inference();
//    //_out->setShape(_in->shape);
//}
//
//string Div::op_to_string() {
//    string result = sub_op_type_string[op] + "<";
//    result+=to_string(divider)+">";
//    return result;
//}
Transform::Transform(TensorHandle _out, TensorHandle in, const vector<Variable> &out_shape, int idx, int count)
    :Computation(_out,TRANSFORM,in,idx),count(count){
    this->out->shape = out_shape;
}

void Transform::compute() {
    Computation::compute();
}

string Transform::to_compute() {
    return Computation::to_compute();
}

string Transform::to_c_statement() {
    return Computation::to_c_statement();
}

void Transform::set_compute() {

}

Reshape::Reshape(TensorHandle _out, TensorHandle _input, int idx,const vector<int>& new_shape) :Computation(_out,RESHAPE,_input,idx),new_shape(new_shape){
    /*
     * this function only set a new shape to tensor, do not change memory layout
     * supported -1 in new_shape
     */
    int size = 1;
    for(const auto& v:_input->shape){
        size *=v.get_upper_bound();
    }
    //Assuming only one -1 at new_shape ,not checking yet
    int new_size=1;
    int neg_index = -1;
    for(int i=0;i<new_shape.size();i++){
        if(new_shape[i]==-1) neg_index = i;
        else new_size*=new_shape[i];
    }
    if(neg_index!=-1) const_cast<int&>(new_shape[neg_index]) = size/new_size; //assuming size%new_size==0
    vector<Variable> vars;
    if(_input->tensorType==INPUT||_input->tensorType==MIX){
        vars = Graph::get_variable(new_shape, true);
    }else{
        vars = Graph::get_variable(new_shape,false);
    }
    // check at Tensor
    _out->reshape(vars);
}

string Reshape::to_compute() {
    string result = this->out->name+" = topi.reshape("+this->in[0]->name+",";
    result+= out->to_string(true,false,false,false,true,false);
    result+=")";
    return result;
}
