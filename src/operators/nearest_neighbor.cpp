//
// Created by sun on 2021/12/13.
//

#include"Ops.h"
#include"Parser.h"
#include<cassert>
NearestNeighbor::NearestNeighbor(TensorHandle _input, const vector<int>& scale, Graph*g): OpBase(_input, OP_UPSAMPLING, g){
    assert(!scale.empty());
    assert(_input->shape.size()==scale.size()+2);
    int length = scale.size();
    vector<int> interpolation;
    interpolation.reserve(length);
    for(int i=0;i<length;i++){
        interpolation.push_back(scale[i]*_input->shape[i+2].get_upper_bound());
    }
    vector<Variable> out_shape;
    if(length>=3) out_shape.push_back(Graph::get_variable(graph->input_d, interpolation[length - 3], 0, 2));//d
    if(length>=2) out_shape.push_back(Graph::get_variable(graph->input_h, interpolation[length - 2], 0, 3));//h
    if(length>=1) out_shape.push_back(Graph::get_variable(graph->input_w, interpolation[length - 1], 0, 4));//w
    auto t0 = new Tensor;
    vector<Affine> lambda0;
    lambda0.emplace_back(_input->shape[0]);
    lambda0.emplace_back(_input->shape[1]);
    for(int i=0;i<length;i++){
        lambda0.push_back(Parser::make_affine({out_shape[i].name, "/", scale[i]}));
    }
    out_shape.insert(out_shape.begin(), _input->shape[0]);
    out_shape.insert(out_shape.begin(), _input->shape[1]);
    //do interpolation
    graph->push(new Assign(t0, _input, out_shape, graph->num_cpt, lambda0, nullptr));//set formula as nullptr, need user make sure affine is legal
    outputs[0] = t0;
}

NearestNeighbor::NearestNeighbor(TensorHandle _input,const vector<int>& numerator,const vector<int>& denominator,Graph* g) :
    OpBase(_input,OP_UPSAMPLING,g){
    /*
     * for float scale : 1.5 ==> 3/2
     */
    assert(!numerator.empty());
    assert(numerator.size()==denominator.size());
    assert(_input->shape.size()==numerator.size()+2);
    int length = numerator.size();
    vector<int> interpolation;
    interpolation.reserve(length);
    for(int i=0;i<length;i++){
        interpolation.push_back(numerator[i]*_input->shape[i+2].get_upper_bound()/denominator[i]);
    }
    vector<Variable> out_shape;
    if(length>=3) out_shape.push_back(Graph::get_variable(graph->input_d, interpolation[length - 3], 0, 2));//d
    if(length>=2) out_shape.push_back(Graph::get_variable(graph->input_h, interpolation[length - 2], 0, 3));//h
    if(length>=1) out_shape.push_back(Graph::get_variable(graph->input_w, interpolation[length - 1], 0, 4));//w
    auto t0 = new Tensor;
    vector<Affine> lambda0;
    lambda0.emplace_back(_input->shape[0]);
    lambda0.emplace_back(_input->shape[1]);
    for(int i=0;i<length;i++){
        lambda0.push_back(Parser::make_affine({out_shape[i].name, "*", denominator[i], "/", numerator[i]}));
    }
    out_shape.insert(out_shape.begin(), _input->shape[0]);
    out_shape.insert(out_shape.begin(), _input->shape[1]);
    graph->push(new Assign(t0, _input, out_shape, graph->num_cpt, lambda0, nullptr));//set formula as nullptr, need user make sure affine is legal
    outputs[0] = t0;
}


