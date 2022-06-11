//
// Created by sun on 2021/12/9.
//

#include"Ops.h"
#include"Parser.h"
#include"Pattern.h"
#include"Producer.h"
#include<cassert>

MaxPooling::MaxPooling(TensorHandle _input,const vector<int>& kernel,const vector<int>& stride,PaddingMode _padding,Graph* g)
        : OpBase(_input, OP_POOLING_MAX, g), kernel(kernel), stride(stride), padding(_padding){
    assert(_input->shape.size()==4);
    assert(kernel.size()==stride.size());
    int inputH = _input->shape[2].get_upper_bound();
    int inputW = _input->shape[3].get_upper_bound();
    int strideH=1,strideW=1,kernelH=1,kernelW=1;
    if(kernel.size()==2){
        strideH = stride[0],strideW = stride[1];
        kernelH = kernel[0],kernelW = kernel[1];
    }
    int outputH, outputW;
    Variable oh,ow;
    int padH,padW;
    switch (padding)
    {
        case PD_MODE_SAME:
            outputH = (inputH + strideH - 1) / strideH;
            outputW = (inputW + strideW - 1) / strideW;
            padH = max(((outputH-1)*strideH+kernelH -inputH)/2,0);
            padW = max(((outputW-1)*strideW+kernelW -inputW)/2,0);
            break;
        case PD_MODE_VALID:
            outputH = (inputH - kernelH) / strideH + 1;
            outputW = (inputW - kernelW) / strideW + 1;
            padH = 0;
            padW =0;
            break;
        default:
            assert(false);
    }
    oh = Graph::get_variable(graph->input_h,outputH,0,3);//h
    ow = Graph::get_variable(graph->input_w,outputW,0,4);//w

    numOutputs = 1;

    if(graph->concat_axis.find(_input->shape[1])==graph->concat_axis.end()){//is that useful
        graph->concat_axis.insert(_input->shape[1]);
    }

    Variable kh,kw;
    if(graph->weight_kh.find(kernelH)==graph->weight_kh.end()){
        graph->weight_kh[kernelH] = Variable(Producer::get_debug_variable_name(1, 3),kernelH);
    }
    kh = graph->weight_kh[kernelH];
    if(graph->weight_kw.find(kernelW)==graph->weight_kw.end()){
        graph->weight_kw[kernelW] = Variable(Producer::get_debug_variable_name(1,4),kernelW);
    }
    kw = graph->weight_kw[kernelW];
    auto t = new Tensor,t1 = new Tensor;
    vector<Affine> lambda;
    lambda.emplace_back(_input->shape[0]);
    lambda.emplace_back(_input->shape[1]);
    map<string,int> eval={make_pair("strideH",strideH),make_pair("strideW",strideW),make_pair("padH",padH),make_pair("padW",padW),
                          make_pair("kernelH",kernelH),make_pair("kernelW",kernelW)};
    map<string,string> var_map = {make_pair("oh",oh.name),make_pair("ow",ow.name)};
    lambda.push_back(Parser::make_affine("oh*strideH+kernelH-padH",var_map,eval));
    lambda.push_back(Parser::make_affine("ow*strideW+kernelW-padW",var_map,eval));
    graph->push(new Assign(t,_input,{_input->shape[0],_input->shape[1],oh,ow,kh,kw},graph->num_cpt,lambda));
    graph->push(new Max(t1,t,{kh,kw},graph->num_cpt));
    outputs[0] = t1;
}
