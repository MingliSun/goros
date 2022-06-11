//
// Created by sun on 2021/11/18.
//

#include"Ops.h"
#include"Parser.h"
#include<cassert>
#include <iostream>
Convolution::Convolution(TensorHandle  _input, TensorHandle  _weight,const vector<int>& stride, PaddingMode _padding, Graph*g)
    : OpBase(_input, _weight, OP_CONVOLUTION, g), stride(stride), padding(_padding){
    /*
     * conv1d conv2d conv3d
     * conv1d : NCW
     * conv2d : NCHW
     * conv3d : NCDHW
     */
    assert(_input->shape.size() == _weight->shape.size());
    int size = _input->shape.size();
    assert(size>= 3);
    assert(_input->shape[1] == _weight->shape[1]);
    vector<int> input_size;
    vector<int> kernel_size;
    vector<Variable> kernel;
    vector<int> output_size;
    int length = size-2;
    for(int i=2;i<size;i++){
        input_size.push_back(_input->shape[i].get_upper_bound());
        kernel_size.push_back(_weight->shape[i].get_upper_bound());
        kernel.push_back(_weight->shape[i]);
    }
    switch (padding)
    {
        case PD_MODE_SAME:
            for(int i=0;i<length;i++){
                output_size.push_back((input_size[i] + stride[i] - 1) / stride[i]);
            }
            break;
        case PD_MODE_VALID:
            for(int i=0;i<length;i++){
                output_size.push_back((input_size[i] - kernel[i].get_upper_bound()) / stride[i] + 1);
            }
            break;
        default:
            assert(false);
    }
    vector<Variable> output;
    if(length>=3) output.push_back(Graph::get_variable(graph->input_d,output_size[length-3],0,2));//d
    if(length>=2) output.push_back(Graph::get_variable(graph->input_h,output_size[length-2],0,3));//h
    if(length>=1) output.push_back(Graph::get_variable(graph->input_w,output_size[length-1],0,4));//w
    vector<int> pad;
    pad.reserve(length);
    for(int i=0;i<length;i++){
        pad.push_back(max(((output_size[i]-1)*stride[i]+kernel_size[i]-input_size[i]+1)/2,0)); // pad at 'front' 'top' 'left' first
    }


    //int outputH = 1 + (inputH + 2 * padH - kernelH) / strideH;
    //int outputW = 1 + (inputW + 2 * padW - kernelW) / strideW;

    if(graph->concat_axis.find(_input->shape[1])==graph->concat_axis.end()){
        graph->concat_axis.insert(_input->shape[1]);
    }

    auto t = new Tensor,t1 = new Tensor ,t2 = new Tensor;
    vector<Affine> lambda;
    lambda.emplace_back(_input->shape[0]);
    lambda.emplace_back(_input->shape[1]);
    for(int i=0;i<length;i++){
        lambda.emplace_back(Parser::make_affine({output[i].name,"*",stride[i],"+",kernel[i].name,"-",pad[i]}));
    }
    vector<Variable> out_shape  = {_input->shape[0],_input->shape[1]};
    for(int i=0;i<length;i++){
        out_shape.push_back(output[i]);
    }
    for(int i=0;i<length;i++){
        out_shape.push_back(kernel[i]);
    }
    kernel.insert(kernel.begin(),_weight->shape[1]);
    graph->push(new Assign(t,_input,out_shape,graph->num_cpt,lambda));
    graph->push(new Mul(t1,t,_weight,graph->num_cpt));
    Sum* s = new Sum(t2,t1,kernel,graph->num_cpt,stride);
    //adjust sum
    graph->adjust_sum(s);
    graph->push(s);
    outputs[0] = t2;
}
GroupConvolution::GroupConvolution(TensorHandle _input, TensorHandle _weight, const vector<int>& stride, PaddingMode _padding, Graph*g)
    :OpBase(_input, _weight, OP_CONVOLUTION, g){
    /*
    * oc =1 and not included in weight->shape
     * 2D n c h w -- c kh kw
    */
    assert(_input->shape.size()== _weight->shape.size());
    int size = _input->shape.size();
    assert(size>= 3);
    assert(_input->shape[1].get_upper_bound() % _weight->shape[1].get_upper_bound()==0);
    int group = _input->shape[1].get_upper_bound() / _weight->shape[1].get_upper_bound();
    assert(_weight->shape[0].get_upper_bound()%group==0);
    int divider = _weight->shape[0].get_upper_bound()/group;
    vector<int> input_size;
    vector<int> kernel_size;
    vector<Variable> kernel;
    vector<int> output_size;
    int length = size-2;
    for(int i=2;i<size;i++){
        input_size.push_back(_input->shape[i].get_upper_bound());
        kernel_size.push_back(_weight->shape[i].get_upper_bound());
        kernel.push_back(_weight->shape[i]);
    }
    switch (_padding)
    {
        case PD_MODE_SAME:
            for(int i=0;i<length;i++){
                output_size.push_back((input_size[i] + stride[i] - 1) / stride[i]);
            }
            break;
        case PD_MODE_VALID:
            for(int i=0;i<length;i++){
                output_size.push_back((input_size[i] - kernel[i].get_upper_bound()) / stride[i] + 1);
            }
            break;
        default:
            assert(false);
    }
    vector<Variable> output;
    if(length>=3) output.push_back(Graph::get_variable(graph->input_d,output_size[length-3],0,2));//d
    if(length>=2) output.push_back(Graph::get_variable(graph->input_h,output_size[length-2],0,3));//h
    if(length>=1) output.push_back(Graph::get_variable(graph->input_w,output_size[length-1],0,4));//w
    vector<int> pad;
    pad.reserve(length);
    for(int i=0;i<length;i++){
        pad.push_back(max(((output_size[i]-1)*stride[i]+kernel_size[i]-input_size[i])/2,0));
    }


    //int outputH = 1 + (inputH + 2 * padH - kernelH) / strideH;
    //int outputW = 1 + (inputW + 2 * padW - kernelW) / strideW;

    if(graph->concat_axis.find(_input->shape[1])==graph->concat_axis.end()){
        graph->concat_axis.insert(_input->shape[1]);
    }

    auto t = new Tensor,t1 = new Tensor ,t2 = new Tensor;
    vector<Affine> lambda;
    lambda.emplace_back(_input->shape[0]);
    lambda.emplace_back(Parser::make_affine({_weight->shape[1].name,"+",_weight->shape[0].name,"/",divider,"*",_weight->shape[1].get_upper_bound()}));
    for(int i=0;i<length;i++){
        lambda.emplace_back(Parser::make_affine({output[i].name,"*",stride[i],"+",kernel[i].name,"-",pad[i]}));
    }
    vector<Variable> out_shape  = {_input->shape[0],_weight->shape[0],_weight->shape[1]};// n oc ic
    out_shape.insert(out_shape.end(),output.begin(),output.end()); // oh ow
    out_shape.insert(out_shape.end(),kernel.begin(),kernel.end());// kh kw
    graph->push(new Assign(t,_input,out_shape,graph->num_cpt,lambda));
    graph->push(new Mul(t1,t,_weight,graph->num_cpt));
    kernel.insert(kernel.begin(),_weight->shape[1]);
    Sum* s = new Sum(t2,t1,kernel,graph->num_cpt,stride); // reduce axis is ic+ [kd,kh,kw]
    //adjust sum
    graph->adjust_sum(s);
    graph->push(s);
    outputs[0] = t2;
}

TransposedConvolution::TransposedConvolution(TensorHandle _input, TensorHandle _weight,const vector<int>& stride, const vector<int>& out_padding, Graph*g)
        : OpBase(_input, _weight, OP_TRANSPOSED_CONVOLUTION, g){
    /*
     * do interpolation first with stride
     * then do convolution with stride=1 and out_padding
     * not a group convolution
     */
    assert(_input->shape.size() == _weight->shape.size());
    assert(stride.size()==out_padding.size()||2*stride.size()==out_padding.size());
    //make sure out_padding < stride, not checking!!
    int size = _input->shape.size();
    assert(size>= 3);
    assert(_input->shape[1] == _weight->shape[1]);
    vector<int> input_size;
    vector<int> kernel_size;
    vector<Variable> kernel;
    int length = size-2;
    for(int i=2;i<size;i++){
        input_size.push_back(_input->shape[i].get_upper_bound());
        kernel_size.push_back(_weight->shape[i].get_upper_bound());
        kernel.push_back(_weight->shape[i]);
    }
    vector<int> interpolation;
    interpolation.reserve(length);
    for(int i=0;i<length;i++){
        interpolation.push_back(input_size[i]+(input_size[i]-1)*(stride[i]-1));
    }
    //do interpolation
    vector<Variable> feature;
    if(length>=3) feature.push_back(Graph::get_variable(graph->input_d, interpolation[length - 3], 0, 2));//d
    if(length>=2) feature.push_back(Graph::get_variable(graph->input_h, interpolation[length - 2], 0, 3));//h
    if(length>=1) feature.push_back(Graph::get_variable(graph->input_w, interpolation[length - 1], 0, 4));//w
    auto t0 = new Tensor;
    vector<Affine> lambda0;
    lambda0.emplace_back(_input->shape[0]);
    lambda0.emplace_back(_input->shape[1]);
    for(int i=0;i<length;i++){
        lambda0.push_back(Parser::make_affine({feature[i].name, "/", stride[i]}));
    }
    vector<Constraint> constraints;
    constraints.reserve(length);
    for(int i=0;i<length;i++){
        constraints.emplace_back(CONSTRAINT_EQ, Parser::make_affine({feature[i].name, "%", stride[i]}), Affine(0));
    }
    auto formula = Formula::create_all(constraints);
    vector<Variable> assign0_shape =  {_input->shape[0],_input->shape[1]};
    assign0_shape.insert(assign0_shape.end(), feature.begin(), feature.end());
    graph->push(new Assign(t0,_input,assign0_shape,graph->num_cpt,lambda0,formula));

    // do conv stride=1 out_padding
    //what if out_padding are not symmetric (left and right) ---done
    vector<int> output_size;
    output_size.resize(length);
    for(int i=0;i<length;i++){
        if(out_padding.size()==stride.size()){
            output_size[i] = 1 + (feature[i].get_upper_bound() + 2 * out_padding[i] - kernel_size[i]) / stride[i];
        }else{ //
            output_size[i] = 1 + (feature[i].get_upper_bound() + out_padding[i*2]+out_padding[i*2+1]- kernel_size[i]) / stride[i];
        }
    }
    vector<Variable> output;
    if(length>=3) output.push_back(Graph::get_variable(graph->input_d,output_size[length-3],0,2));//d
    if(length>=2) output.push_back(Graph::get_variable(graph->input_h,output_size[length-2],0,3));//h
    if(length>=1) output.push_back(Graph::get_variable(graph->input_w,output_size[length-1],0,4));//w
    vector<int> pad;
    for(int i=0;i<length;i++){
        if(out_padding.size()==stride.size()){
            pad.push_back(out_padding[i]);
        }else{
            pad.push_back(out_padding[2*i]);
        }
    }

    //int outputH = 1 + (inputH + 2 * padH - kernelH) / strideH;
    //int outputW = 1 + (inputW + 2 * padW - kernelW) / strideW;

    if(graph->concat_axis.find(_input->shape[1])==graph->concat_axis.end()){
        graph->concat_axis.insert(_input->shape[1]);
    }

    auto t = new Tensor,t1 = new Tensor ,t2 = new Tensor;
    vector<Affine> lambda;
    lambda.emplace_back(_input->shape[0]);//n
    lambda.emplace_back(_input->shape[1]);//c
    for(int i=0;i<length;i++){
        lambda.emplace_back(Parser::make_affine({output[i].name,"+",kernel[i].name,"-",pad[i]}));
    }
    vector<Variable> out_shape  = {_input->shape[0],_input->shape[1]}; //n c
    out_shape.insert(out_shape.end(),output.begin(),output.end());// oh ow
    out_shape.insert(out_shape.end(),kernel.begin(),kernel.end());//kh kw
    kernel.insert(kernel.begin(),_weight->shape[1]);//regard as reduce_axis
    graph->push(new Assign(t,t0,out_shape,graph->num_cpt,lambda));
    graph->push(new Mul(t1,t,_weight,graph->num_cpt));
    Sum* s = new Sum(t2,t1,kernel,graph->num_cpt,stride);
    //adjust sum
    graph->adjust_sum(s);
    graph->push(s);
    outputs[0] = t2;
}