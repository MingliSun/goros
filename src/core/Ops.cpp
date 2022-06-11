//
// Created by sun on 2021/11/18.
//
#include"Ops.h"
#include"Pattern.h"
#include"Producer.h"
#include<cassert>
#include<iostream>
#include<fstream>
#include <utility>

map<int,Variable> Graph::weight_oc;
map<int,Variable> Graph::channel;
map<int,Variable> Graph::weight_kd;//for 3d
map<int,Variable> Graph::weight_kh;
map<int,Variable> Graph::weight_kw;
map<int,Variable> Graph::input_n;
map<int,Variable> Graph::input_d;//for 3d
map<int,Variable> Graph::input_h;
map<int,Variable> Graph::input_w;
map<int,Variable> Graph::other;


OpBase::OpBase(TensorHandle _input, OpType _type,Graph* _graph) :numInputs(1),type(_type),graph(_graph){
    inputs[0] = _input;
    for (auto & output : outputs) {
        output= nullptr;
    }
}

OpBase::OpBase(TensorHandle _input0, TensorHandle _input1, OpType _type,Graph* _graph):numInputs(2),type(_type),graph(_graph) {
    inputs[0] = _input0;
    inputs[1] = _input1;
    for (auto & output : outputs) {
        output = nullptr;
    }
}

OpBase::OpBase(TensorHandle input0, TensorHandle input1, TensorHandle input2, OpType _type,Graph* _graph) :numInputs(3),type(_type),graph(_graph) {
    inputs[0] = input0;
    inputs[1] = input1;
    inputs[2] = input2;
    for (auto & output : outputs) {
        output= nullptr;
    }
}

OpBase::OpBase(TensorHandle input0, TensorHandle input1, TensorHandle input2, TensorHandle input3, OpType _type,Graph* _graph):numInputs(4),type(_type) ,graph(_graph){
    inputs[0] = input0;
    inputs[1] = input1;
    inputs[2] = input2;
    inputs[3] = input3;
    for (auto & output : outputs) {
        output= nullptr;
    }
}

OpBase::OpBase(TensorHandle input0, TensorHandle input1, TensorHandle input2, TensorHandle input3,
               TensorHandle input4, OpType _type,Graph* _graph):numInputs(5),type(_type) ,graph(_graph){
    inputs[0] = input0;
    inputs[1] = input1;
    inputs[2] = input2;
    inputs[3] = input3;
    inputs[4] = input4;
    for (auto & output : outputs) {
        output= nullptr;
    }
}

OpBase::OpBase(int n, TensorHandle *_inputs, OpType _type,Graph* _graph) :numInputs(n),type(_type),graph(_graph){
    assert(n <= MAX_NUM_INPUTS);
    for (int i = 0; i < n; i++)
        inputs[i] = _inputs[i];
    for (auto & output : outputs) {
        output= nullptr;
    }
}

OpBase::OpBase(vector<TensorHandle> _inputs,OpType _type,Graph* _graph):numInputs(_inputs.size()),type(_type),graph(_graph) {
    int n = _inputs.size();
    assert(n <= MAX_NUM_INPUTS);
    for (int i = 0; i < n; i++)
        inputs[i] = _inputs[i];
    for (auto & output : outputs) {
        output= nullptr;
    }
}


TensorHandle Graph::new_input(const vector<int>& shape,DataType _dataType,string name) {
    vector<Variable> vars  = get_variable(shape);
    auto res =  new Tensor(vars,INPUT,_dataType, nullptr,std::move(name));
    inputs.push_back(res);
    return res;
}

TensorHandle Graph::new_weight(const vector<int> &shape, DataType dataType,string name,bool depthwise) {
    vector<Variable> vars;
    vars = get_variable(shape,false,depthwise);
    void* _ptr;
    auto sdt = Tensor::convert_data_type(dataType);
    switch (sdt) {
        case SDT_FLOAT:
            _ptr = Producer::allocate_sequential<float>(vars);
            break;
        case SDT_INT:
            _ptr = Producer::allocate_sequential<int>(vars);
            break;
        case SDT_UINT:
            _ptr = Producer::allocate_sequential<uint>(vars);
            break;
        case SDT_INVALID:
            assert(false);
            break;
    }
    return new Tensor(vars,WEIGHT,dataType,_ptr,std::move(name));
}

void Graph::push(Computation *c) {
    this->computations[num_cpt++] = c;
//    real_count = num_cpt;
//    frontIdx = num_cpt - 1;
}

Graph::Graph() = default;

TensorHandle Graph::conv2d(TensorHandle _input, TensorHandle _weight, int _strideH, int _strideW, PaddingMode _padding) {
//    int oc = _weight->shape[0].get_upper_bound(),ic=_weight->shape[1].get_upper_bound();
//    int h = _input->shape[2].get_upper_bound(),w = _input->shape[3].get_upper_bound();
//    int kh = _weight->shape[2].get_upper_bound(),kw=_weight->shape[3].get_upper_bound();
//    if(kh==3&&kw==3&&_strideH==1&&_strideW==1&&oc>=16&&ic>=16&&h<=120&&w<=120){
//        auto item = new Winograd(_input,_weight,2,_padding,this);
//        return item->outputs[0];
//    }
    auto item = new Convolution(_input, _weight, {_strideH,_strideW}, _padding, this);
    return item->outputs[0];
}

string Graph::to_string() {
    string result;
    for(int i=0;i<num_cpt;i++){
        if(computations[i]!= nullptr){
            result+= computations[i]->out->to_string(false,false,true,true,true, true)+" = ";
            result+= computations[i]->op_to_string();
            result+="(";
            for(int j=0;j<computations[i]->num_in;j++){
                result+=computations[i]->in[j]->to_string();
                if(j!=computations[i]->num_in-1) result+=",";
            }
            result+=")";
            if(computations[i]->out->data_ptr) result+=" has value";
            else result+= " no value,";
            if(computations[i]->out->group_id.empty()) result+=" no group";
            else result+=" group "+set_to_string(computations[i]->out->group_id);
            result+="\n";
        }
    }
    return result;
}

TensorHandle Graph::element(TensorHandle  _left,TensorHandle  _right,OpType _type) {
    auto * ew = new Element(_left,_right,_type,this);
    return ew->outputs[0];
}

TensorHandle Graph::batch_norm(TensorHandle _input, TensorHandle _scale, TensorHandle _bias, TensorHandle _mean,
                               TensorHandle _var, float _epsilon) {
    auto * bn = new BatchNorm(_input,_scale,_bias,_mean,_var,_epsilon,this);
    return bn->outputs[0];
}

void Graph::optimize() {
    auto *p = new PatternMatch(this);
    p->optimize();
}

Compare Graph::is_single_concat_axis(const vector<Variable>& var0s,const vector<Variable>& var1s,Variable& v0,Variable& v1) {
    if(var0s.size()!=var1s.size()) return COMPARE_ER;
    int n = var0s.size();

    int count0=0,count1=0,index0=0,index1=0;

    for(int i=0;i<n;i++){
        if(concat_axis.find(var0s[i])!=concat_axis.end()){
            v0 = var0s[i];
            count0++;
            index0=i;
        }
        if(concat_axis.find(var1s[i])!=concat_axis.end()){
            v1 = var1s[i];
            count1++;
            index1=i;
        }
    }
    if(count0!=1||count1!=1) return COMPARE_ER;
    if(index0!=index1) return COMPARE_ER;
    bool flag = true;
    for(int i=0;i<n;i++){
        if(i==index0) continue;
        if(var0s[i]!=var1s[i]) {flag = false;break;}
    }
    if(flag) return COMPARE_EQ;
    flag = true;
    for(int i=0;i<n;i++){
        if(i==index0) continue;
        if(var0s[i].get_upper_bound()>=var1s[i].get_upper_bound()) {flag = false;break;}
    }
    if(flag) return COMPARE_LT;
    flag = true;
    for(int i=0;i<n;i++){
        if(i==index0) continue;
        if(var0s[i].get_upper_bound()<=var1s[i].get_upper_bound()) {flag = false;break;}
    }
    if(flag) return COMPARE_GT;
    return COMPARE_ER;
}

//TensorHandle Graph::new_weight(const vector<Variable>& shape, void*_ptr,const string& name) {
//    if(name.empty())
//        return new Tensor(shape, WEIGHT, _ptr, Producer::get_unique_tensor_name());
//    else
//        return new Tensor(shape,WEIGHT,_ptr,name);
//}

void Graph::codegen_te(const string& filename,int trials,bool resume,const string& target,const string& host) {
    // tvm 0.9dev
    ofstream file;
    file.open(filename);
    if(resume) file<<"from resume_research import resume_research"<<endl;// a common function
    file<<"import numpy as np"<<endl;
    file<<"import tvm"<<endl;
    file<<"from tvm import te, auto_scheduler,topi"<<endl;
    for(auto & fused_operator : fused_operators){
        if(fused_operator.computations.empty()) continue;
        file<<"@auto_scheduler.register_workload"<<endl;
        file<<"def "<<fused_operator.name<<"():"<<endl;
        fused_operator.generate_te();
        for(auto& p:fused_operator.source_te){
            file<<"\t"<<p<<endl;
        }
    }
    file<<"target = tvm.target.Target('"<<target<<"',host='"<<host<<"')"<<endl;
    if(target=="cuda") file<<"dev = tvm.cuda()"<<endl;
    else file<<"dev=tvm.cpu()"<<endl;
    //special case zero
    file<<"zero= tvm.nd.array(np.array(0).astype(np.float32),dev)"<<endl;
    for(auto ri =fused_operators.rbegin();ri!=fused_operators.rend();ri++){
        auto fused_operator = *ri;
        if(fused_operator.computations.empty()) continue;
        string log_file =fused_operator.name+".json" ;
        file<<"task = tvm.auto_scheduler.SearchTask(func="<<fused_operator.name<<", args=(), target=target)"<<endl;
        if(resume){
            file<<"resume_search(task,'"<<log_file<<"',"<<trials<<")"<<endl;
        }else{
            file<<"tune_option= auto_scheduler.TuningOptions(num_measure_trials="<<trials;
            file<<", measure_callbacks=[auto_scheduler.RecordToFile('"<<log_file<<"')],verbose=0,)"<<endl;
            file<<"task.tune(tune_option)"<<endl;
        }
        file<<"sch, args = task.apply_best('"<<log_file<<"')"<<endl;
        file<<fused_operator.name<<"_func = tvm.build(sch,args,target)"<<endl;
        //evaluator for random data
        string signature;
        for(auto & i : fused_operator.signature){
            signature+= i->name+",";
            if(i->name=="zero"|| i->is_used_for_te) continue; // different Mechanism to make sure that only print same tensor one time,it seems in disorder.
            if(i->sub_op_type!=DATA&&i->tensorType!=WEIGHT&&!is_element_in_set(i->group_id,fused_operator.id)) continue;//feature map from other subgraph
            if(i->is_boundary) file<<i->name<<"= tvm.nd.array(np.zeros("<<i->to_string(true,false,false,false,true)<<",dtype='float32'),dev)"<<endl;
            else file<<i->name<<"= tvm.nd.array(np.random.uniform(size="<<i->to_string(true,false,false,false,true)<<").astype(np.float32),dev)"<<endl;
            i->is_used_for_te = true;
        }

        file<<fused_operator.name<<"_func("<<signature.substr(0,signature.size()-1)<<")"<<endl;
        file<<"evaluator = "<<fused_operator.name<<"_func.time_evaluator("<<fused_operator.name<<"_func.entry_name,dev,min_repeat_ms=500)"<<endl;
        file<<"print('Execution time of "<<fused_operator.name<<": %.3fms'%(np.median(evaluator("<<signature.substr(0,signature.size()-1)
            <<").results)*1000))"<<endl;
    }

    file.close();
}

void Graph::codegen_c(const string& filename) {
    ofstream file;
    file.open(filename);
    //Assuming graph share same data type ,if not,add it to  todo list
    string t = supported_data_type_string[Tensor::convert_data_type(computations[frontIdx]->out->dataType)]+" ";
    for(auto & fused_operator : fused_operators){
        file<<"void "<<fused_operator.name<<"(";
        for(int j=0;j<fused_operator.signature.size();j++){

            file<<t<<fused_operator.signature[j]->to_string(true,true);
            if(j!=fused_operator.signature.size()-1) file<<",";
        }
        file<<"){"<<endl;//1
        file<<"#pragma scop"<<endl;
        fused_operator.generate_c();
        assert(fused_operator.c_loops.size()==fused_operator.c_statements.size());//c_loops equals c_statements at size prob
        for(auto & j : fused_operator.intermediate_variable){//intermediate_variable
            file<<"\t"<<t<<j->to_string(true,true)<<";"<<endl;
        }
        for(int j=0;j<fused_operator.c_loops.size();j++){
            for(auto & k : fused_operator.c_loops[j]){
                file<<"\t"<<k.to_string()<<"{"<<endl;
            }
            for(auto & k : fused_operator.c_statements[j]){
                file<<"\t\t"<<k<<endl;
            }
            for(int k=0;k<fused_operator.c_loops[j].size();k++){
                file<<"\t}";
            }
            file<<endl;
        }
        file<<"#pragma endscop"<<endl<<"}"<<endl;
    }
    file.close();
}

void Graph::codegen_dot(const string& filename) {
    ofstream file;
    file.open(filename);
    file<<"digraph demo{"<<endl;//todo graph name can distract from filename
    for(int i=int(fused_operators.size()-1);i>=0;i--){
        file<<"\tsubgraph cluster_"<<::to_string(i)<<"{"<<endl;
        file<<"\t\tnode[style=filled];"<<endl;
        file<<"\t\tlabel=\""<<fused_operators[i].name<<"\";"<<endl;
        file<<"\t\tcolor=blue;"<<endl;
        fused_operators[i].generate_dot();
        for(auto& p:fused_operators[i].source_dot){
            file<<"\t\t"<<p<<endl;
        }
        file<<"\t}"<<endl;
    }
    file<<"}"<<endl;
    file.close();
}

void Graph::print_weights(const string& filename) {
    ofstream file;
    file.open(filename);

    for(auto&p:computations){
        if(p&&p->in[0]->tensorType==WEIGHT) p->in[0]->print_data(file);
        if(p&&p->num_in>1&&p->in[1]->tensorType==WEIGHT) p->in[1]->print_data(file);
    }

    file.close();
}

TensorHandle Graph::relu(TensorHandle _input) {
    /*
     * remove class Relu ,write here ---done
     */
    auto t = new Tensor;
    push(new SMax(t,_input,&Tensor::zero,num_cpt));
    return t;
}

TensorHandle Graph::max_pool2d(TensorHandle _input, int _kernelH, int _kernelW, int strideH, int strideW, PaddingMode _padding) {
    auto* item = new MaxPooling(_input, {_kernelH, _kernelH}, {strideH, strideW}, _padding, this);
    return item->outputs[0];
}

void Graph::adjust_sum(Sum* sum) {
    if(channel.find(sum->out->shape[1].get_upper_bound()) != channel.end()) {//c
        auto v = channel[sum->out->shape[1].get_upper_bound()];
        sum->subst = make_pair(v,sum->out->shape[1]);
        sum->out->shape[1] = v;
    }else{
        auto v = Graph::get_variable(channel, sum->out->shape[1].get_upper_bound(), 0, 1);//c
        sum->subst = make_pair(v,sum->out->shape[1]);
        sum->out->shape[1] = v;
    }
}

TensorHandle Graph::dropout(TensorHandle _input,float rate) {
    auto item = new Dropout(_input,rate,this);
    return item->outputs[0];
}

Variable Graph::get_variable(map<int,Variable>& m,int value,int row,int column) {
    if(m.find(value)==m.end()) {
        m[value] = Variable( Producer::get_debug_variable_name(row,column),value);
    }
    return m[value];
}
Variable Graph::get_variable(int axis,int value,int size) {
    switch (axis) {
        case 1:
            return Graph::get_variable(input_n,value,0,0);
        case 2:
            return Graph::get_variable(channel,value,0,1);
        case 3:
            if(size==3) return Graph::get_variable(input_w,value,0,4);
            if(size==4) return Graph::get_variable(input_h,value,0,3);
            if(size==5) return Graph::get_variable(input_d,value,0,2);
        case 4:
            if(size==4) return Graph::get_variable(input_w,value,0,4);
            if(size==5) return Graph::get_variable(input_h,value,0,3);
        case 5:
            if(size==5) return Graph::get_variable(input_w,value,0,4);
        default:
            return Variable(Producer::get_unique_variable_name());
    }
}

TensorHandle
Graph::conv2d_transpose(TensorHandle _input, TensorHandle _weight, int _strideH, int _strideW, const vector<int>& out_padding)  {
    auto item = new TransposedConvolution(_input, _weight,{ _strideH, _strideW},out_padding,this);
    return item->outputs[0];
}

TensorHandle Graph::concat(const vector<TensorHandle>& input, int axis) {
    auto item = new Concatenate(input, axis, this);
    return item->outputs[0];
}

TensorHandle Graph::upsampling(TensorHandle _input, const base& scale_h, const base& scale_w,const string& method) {
    if(method=="nearest_neighbor"){
        if(scale_h.index()==1||scale_w.index()==1){//one of scale is float
            int numerator_h, denominator_h=1;
            int numerator_w, denominator_w=1;
            float2fraction(get<float>(scale_h),numerator_h,denominator_h);
            float2fraction(get<float>(scale_w),numerator_w,denominator_w);
            auto item = new NearestNeighbor(_input,{numerator_h,numerator_w},{denominator_h,denominator_w},this);
            return item->outputs[0];
        }else{//all int
            vector<int> scale;
            scale.push_back(get<int>(scale_h));
            scale.push_back(get<int>(scale_w));
            auto item = new NearestNeighbor(_input,scale,this);
            return item->outputs[0];
        }
    }
    throw exception();
}

TensorHandle Graph::sigmoid(TensorHandle _input) {
    auto item = new Sigmoid(_input,this);
    return item->outputs[0];
}

TensorHandle Graph::add(TensorHandle  _left,TensorHandle  _right) {
    auto item = new Element(_left,_right,OP_EW_ADD,this);
    return item->outputs[0];
}

TensorHandle Graph::multiply(TensorHandle  _left, TensorHandle  _right) {
    auto item = new Element(_left,_right,OP_EW_MUL,this);
    return item->outputs[0];
}

TensorHandle
Graph::avg_pool2d(TensorHandle _input, int _kernelH, int _kernelW, int strideH, int strideW, PaddingMode _padding) {
    auto item = new AvgPooling(_input,{_kernelH,_kernelW},{strideH,strideW},_padding,this);
    return item->outputs[0];
}

TensorHandle Graph::global_avg_pool2d(TensorHandle _input) {
    int kernelH = _input->shape[2].get_upper_bound();
    int kernelW = _input->shape[3].get_upper_bound();
    auto item = new AvgPooling(_input,{kernelH,kernelW},{1,1},PD_MODE_VALID,this);
    return item->outputs[0];
}

TensorHandle Graph::global_max_pool2d(TensorHandle _input) {
    int kernelH = _input->shape[2].get_upper_bound();
    int kernelW = _input->shape[3].get_upper_bound();
    auto item = new MaxPooling(_input,{kernelH,kernelW},{1,1},PD_MODE_VALID,this);
    return item->outputs[0];
}

TensorHandle Graph::batch_flatten(TensorHandle _input) {
    auto item = new BatchFlatten(_input,this);
    return item->outputs[0];
}

TensorHandle Graph::dense(TensorHandle _input, TensorHandle _weight) {
    auto item = new Dense(_input, _weight, this);
    return item->outputs[0];
}

TensorHandle Graph::bias_add(TensorHandle _input, TensorHandle _bias) {
    assert(_bias->shape.size()==1);
    auto t = new Tensor;
    this->push(new Add(t,_input,_bias,this->num_cpt));
    return t;
}

TensorHandle Graph::new_weight(int value,int axis,DataType _dataType,string name) {
    /*
     * for 2D, is that useful?
     */
    vector<Variable> vars;
    switch (axis) {
        case 0:
            vars.push_back(Graph::get_variable(input_n,value,0,0));
            break;
        case 1:
            vars.push_back(Graph::get_variable(channel, value, 0, 1));
            break;
        case 2:
            vars.push_back(Graph::get_variable(input_h,value,0,2));
            break;
        case 3:
            vars.push_back(Graph::get_variable(input_w,value,0,3));
            break;
        default:
            assert(false);
            break;
    }
    void* _ptr;
    auto sdt = Tensor::convert_data_type(_dataType);
    switch (sdt) {

        case SDT_FLOAT:
            _ptr = Producer::allocate_sequential<float>(vars);
            break;
        case SDT_INT:
            _ptr = Producer::allocate_sequential<int>(vars);
            break;
        case SDT_UINT:
            _ptr = Producer::allocate_sequential<uint>(vars);
            break;
        case SDT_INVALID:
            assert(false);
            break;
    }
    return new Tensor(vars,WEIGHT,_dataType,_ptr,std::move(name));
}

TensorHandle Graph::softmax(TensorHandle _input,int axis) {
    /*
     * support negative -1 -2 -3
     */
    Variable v;
    if(axis<0){
        v = _input->shape[_input->shape.size()+axis];
    }else{
        v = _input->shape[axis];
    }
    auto item = new Softmax(_input,v,this);
    return item->outputs[0];
}

TensorHandle
Graph::conv3d(TensorHandle _input, TensorHandle _weight, int _strideD,int _strideH, int _strideW, PaddingMode _padding) {
    auto item = new Convolution(_input, _weight, {_strideD,_strideH,_strideW}, _padding, this);
    return item->outputs[0];
}

void Graph::function(const vector<TensorHandle> &net) {
    outputs = net;
}

TensorHandle Graph::reshape(TensorHandle _input,const vector<int>& new_shape) {
    auto t = new Tensor;
    push(new Reshape(t,_input,num_cpt,new_shape));
    return t;
}

vector<Variable> Graph::get_variable(const vector<int>& shape,bool input,bool depthwise) {
    vector<Variable> vars;
    if(input){
        switch (shape.size()) {
            case 1: //vector
                vars.push_back(get_variable(input_n,shape[0],0,0));
                break;
            case 2://matrix
                vars.push_back(get_variable(input_n,shape[0],0,0));
                vars.push_back(get_variable(channel, shape[1], 0, 1));
                break;
            case 3:// NCW
                vars.push_back(get_variable(input_n,shape[0],0,0));
                vars.push_back(get_variable(channel, shape[1], 0, 1));
                vars.push_back(get_variable(input_w,shape[2],0,4));
                break;
            case 4:{//NCHW
                vars.push_back(get_variable(input_n,shape[0],0,0));
                vars.push_back(get_variable(channel, shape[1], 0, 1));
                vars.push_back(get_variable(input_h,shape[2],0,3));
                vars.push_back(get_variable(input_w,shape[3],0,4));
                break;
            }
            case 5:{//NCDHW
                vars.push_back(get_variable(input_n,shape[0],0,0));
                vars.push_back(get_variable(channel, shape[1], 0, 1));
                vars.push_back(get_variable(input_d,shape[2],0,2));
                vars.push_back(get_variable(input_h,shape[3],0,3));
                vars.push_back(get_variable(input_w,shape[4],0,4));
                break;
            }
            default:
                for(int i:shape){
                    vars.emplace_back(Producer::get_unique_variable_name());
                }
                break;
        }
    }else{
        switch (shape.size()) {
            case 1: //bias vector
                vars.push_back(get_variable(channel,shape[0],0,1));//c
                break;
            case 2: //matrix
                vars.push_back(get_variable(channel, shape[0], 0, 1)); //c
                vars.push_back(get_variable(weight_kw,shape[1],1,4));//kw

                break;
            case 3://OIW
                vars.push_back(get_variable(channel,shape[0],0,1));//c
                vars.push_back(get_variable(weight_kh, shape[1], 1, 3));//kh
                vars.push_back(get_variable(weight_kw,shape[2],1,4));//kw
                break;
            case 4:{
                if(depthwise){
                    vars.push_back(get_variable(channel, shape[0], 0, 1)); //c
                    vars.push_back(get_variable(weight_kd,shape[1],1,2));//kd
                    vars.push_back(get_variable(weight_kh,shape[2],1,3));//kh
                    vars.push_back(get_variable(weight_kw,shape[3],1,4));//kw
                }else{
                    vars.push_back(get_variable(weight_oc,shape[0],1,0));//oc
                    vars.push_back(get_variable(channel, shape[1], 0, 1)); //c
                    vars.push_back(get_variable(weight_kh,shape[2],1,3));//kh
                    vars.push_back(get_variable(weight_kw,shape[3],1,4));//kw
                }
                break;
            }
            case 5:
                vars.push_back(get_variable(weight_oc,shape[0],1,0));//oc
                vars.push_back(get_variable(channel, shape[1], 0, 1)); //c
                vars.push_back(get_variable(weight_kd,shape[2],1,2));//kd
                vars.push_back(get_variable(weight_kh,shape[3],1,3));//kh
                vars.push_back(get_variable(weight_kw,shape[4],1,4));//kw
            default:
                for(int i:shape){
                    vars.emplace_back(Producer::get_unique_variable_name());
                }
                break;
        }
    }

    return vars;
}

TensorHandle Graph::leaky_relu(TensorHandle _input, float alpha) {
    auto alpha_tensor = new Tensor({},WEIGHT,DT_FLOAT,new float{alpha},"alpha");
    auto t = new Tensor,t1 = new Tensor;
    this->push(new Mul(t,_input,alpha_tensor,num_cpt));
    this->push(new SMax(t1,_input,t,this->num_cpt));
    return t1;
}

TensorHandle Graph::yolo_reorg(TensorHandle _input, int stride) {
    auto item = new YoloReorg(_input,stride,this);
    return item->outputs[0];
}

void Graph::split(TensorHandle input, int indices, int axis, vector<TensorHandle>& output) {
    vector<int> sections;
    int size  =input->shape[axis].get_upper_bound();
    assert(size%indices==0);
    int num = 0;
    for(int i=0;i<size-1;i++){
        num+= size/indices;
        sections.push_back(num);
    }
    auto item = new Split(input,sections,axis,this);
    assert(output.size()==item->numOutputs);
    for(int i=0;i<item->numOutputs;i++){
        output[i] = item->outputs[i];
    }
}

void Graph::split(TensorHandle input, const vector<int> &sections, int axis,vector<TensorHandle>& output) {
    auto item = new Split(input,sections,axis,this);
    assert(output.size()==item->numOutputs);
    for(int i=0;i<item->numOutputs;i++){
        output[i] = item->outputs[i];
    }
}

TensorHandle
Graph::conv2d_group(TensorHandle _input, TensorHandle _weight, int _strideH, int _strideW, PaddingMode _padding) {
    /*
     * for now, better using conv2d or using conv2d_group only, do not mix it
     */
    auto item = new GroupConvolution(_input, _weight, {_strideH,_strideW}, _padding, this);
    return item->outputs[0];
}

TensorHandle Graph::tanh(TensorHandle _input) {
    auto item = new Tanh(_input,this);
    return item->outputs[0];
}

TensorHandle Graph::log(TensorHandle _input) {
    /*
     *  x ==> In(1+e^x)
     */
    auto t = new Tensor,t1 = new Tensor,t2 = new Tensor;
    this->push(new Exp(t,_input,num_cpt));
    this->push(new Add(t1,t,&Tensor::one,num_cpt));
    this->push(new Log(t2,t1,num_cpt));
    return t2;
}

TensorHandle Graph::pad(TensorHandle _input, const vector<int> &pad_axis, const vector<int> &pad_value) {
    auto item = new Pad(_input,pad_axis,pad_value,this);
    return item->outputs[0];
}

TensorHandle Graph::slice(TensorHandle _input, const vector<int> &starts, const vector<int> &ends, const vector<int> &axes,
             const vector<int> &steps) {
    auto item = new Slice(_input,starts,ends,axes,steps,this);
    return item->outputs[0];
}

void Graph::float2fraction(float target, int &numerator, int &denominator) {
    /*
     * positive finite  float(decimal) to fraction
     * if it is infinite  recurring float(decimal) ,todo
     */
    assert(target>=0);
    denominator = 1;
    while(target- int(target)>1e-5){// cause target>=0 ,we don't need abs()
        target*=10;
        denominator*=10;
    }
    numerator = int(target);
    int gcd = Math::greatest_common_divisor(numerator,denominator);
    numerator/=gcd;
    denominator/=gcd;
}

TensorHandle Graph::transpose(TensorHandle _input,const vector<int> &perm) {
    vector<int> p;
    if(perm.empty()){
        for(int i=_input->shape.size()-1;i>=0;i--) p.push_back(i);
    }else{
        p = perm;
    }
    auto item = new Transpose(_input,p,this);
    return item->outputs[0];
}

string Graph::set_to_string(const set<int>& s) {
    string ans;
    for(int i : s){
        ans+=std::to_string(i)+" ";
    }
    return ans;
}

bool Graph::is_element_in_set(const set<int>& s,int element) {
    return s.find(element)!=s.end();
//    for(auto& item:s){
//        if(element==item) return true;
//    }
//    return false;
}

void FusedOperator::push(Computation *c) {
    this->computations.push_back(c);
}

void FusedOperator::generate_te() {
    //assuming any two tensors are not same
    //assuming calling order is sequential
    signature.clear();
    int zero_count=0;
    for(int i=0;i<computations.size();i++){
        auto ptr = computations[i];
        for(int j=0;j<ptr->num_in;j++){
            auto tensor = ptr->in[j];
            // make sure one zero in function
            if(tensor->name=="zero") zero_count++;// todo deprecated at next version
            if(zero_count>1||tensor->is_used_for_te) continue;
            if (tensor->tensorType==WEIGHT|| tensor->sub_op_type == DATA|| tensor->group_id.find(id) == tensor->group_id.end()){// not found
                source_te.push_back(ptr->in[j]->to_placeholder());
                signature.push_back(ptr->in[j]);
                tensor->is_used_for_te = true;
            }
        }
        //Assuming last computation is one of last output,cause initializing all computation->is_boundary is complex,we can add it to a todo
        if((i==computations.size()-1)||ptr->is_boundary){
            signature.push_back(ptr->out);
            ptr->out->is_boundary = true;
        }
        //reduce type
        if(ptr->op==SUM){// reduce_axis
            auto item = dynamic_cast<Sum*>(ptr);
            for(auto&p:item->to_reduce_axis()){
                source_te.push_back(p);
            }
        }else if(ptr->op==MAX){
            auto item = dynamic_cast<Max*>(ptr);
            for(auto&p:item->to_reduce_axis()){
                source_te.push_back(p);
            }
        }
        //combine mul and it's consumer
        //todo if mul has two or more consumer,do not combine,it that useful?
        if(ptr->op==MUL&&!ptr->is_boundary) ptr->set_compute();
        else source_te.push_back(ptr->to_compute());
    }
    // initial back tensor.is_used_for_te
    for(auto ptr:computations){
        for(int j=0;j<ptr->num_in;j++) {
            auto tensor = ptr->in[j];
            tensor->is_used_for_te = false;
        }
    }

    string result = "return[";
    for(auto iter = signature.begin();iter!=signature.end();iter++) {
        result+=(*iter)->name;
        if(iter!=signature.end()-1) result+=",";
    }
    result+="]";
    source_te.push_back(result);
}

FusedOperator::FusedOperator() {
    this->name = Producer::get_fused_operator_name();
}

void FusedOperator::generate_dot() {
    for(int i=0;i<computations.size();i++){
        auto ptr = computations[i];
        string node = ptr->out->name+"[label=\""+ptr->op_to_string()+"\"];" ;
        source_dot.push_back(node);
        for(int j=0;j<ptr->num_in;j++){
            if (ptr->in[j]->tensorType==WEIGHT|| ptr->in[j]->sub_op_type == DATA){
                string weight = ptr->in[j]->name+"[shape=plaintext];";
                source_dot.push_back(weight);
            }
            //do not show zero in the dot, if you want to show it, eliminate if statement
            if(ptr->in[j]!=&Tensor::zero) source_dot.push_back(ptr->in[j]->name+"->"+ptr->out->name+";");
        }
    }
}

void FusedOperator::generate_c() {
    signature.clear();
    for(int i=0;i<computations.size();i++){
        auto ptr = computations[i];
        for(int j=0;j<computations[i]->num_in;j++){
            if (i==0||ptr->in[j]->tensorType==WEIGHT|| ptr->in[j]->sub_op_type == DATA){signature.push_back(ptr->in[j]); }
            else intermediate_variable.push_back(ptr->in[j]);
        }
        if(i==computations.size()-1) signature.push_back(ptr->out);
    }

    for(int i=0;i<computations.size();i++){
        auto ptr = computations[i];
        bool found= false;
        vector<Variable> tmp;
        if(ptr->op==SUM) tmp = ptr->in[0]->shape;
        else tmp = ptr->out->shape;
        for(int j=0;j<c_loops.size();j++){//search
            if(is_contain(tmp,c_loops[j])){//assuming sum in[0].shape has been appended
                found = true;
                c_loops[j] = tmp;
                c_statements[j].push_back(ptr->to_c_statement());
                break;
            }
        }
        if(!found){
            c_loops.push_back(ptr->out->shape);
            c_statements.push_back({ptr->to_c_statement()});
        }
        if(computations[i]->op==SUM){
            auto sum = dynamic_cast<Sum*>(computations[i]);
            c_loops.insert(c_loops.begin(),sum->get_original_shape());
            c_statements.insert(c_statements.begin(),{sum->to_c_statement_initial()});
        }
    }

}

bool FusedOperator::is_contain(const vector<Variable>& big,const vector<Variable>& small) {
    int i=0,j=0;
    for(;i<small.size();i++){
        bool found = false;
        for(;j<big.size();j++){
            if(small[i]==big[j]){found = true;j++;break;}
        }
        if(!found) return false;
    }
    return true;
}
