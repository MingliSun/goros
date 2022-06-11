//
// Created by sun on 2021/12/29.
//

#include"Ops.h"

YoloReorg::YoloReorg(TensorHandle _input,int stride,Graph* g): //on need to store stride like used to be
    OpBase(_input,OP_WHERE,g){
     /*
      * only support 4D tensor layout:NCHW, todo support 3D 4D 5D with different stride
      * example : (1,4,2,2) with stride=2 ==> (1,16,1,1)
      */
    assert(_input->shape.size()==4);
    vector<Affine> lambda;
    vector<Variable> out_shape;
    int c = _input->shape[1].get_upper_bound();//4
    int h  = _input->shape[2].get_upper_bound();//2
    int w  = _input->shape[3].get_upper_bound();//2
    int s2 = stride*stride;
    out_shape.push_back(_input->shape[0]);//n
    lambda.emplace_back(_input->shape[0]);//n
    auto oc = Graph::get_variable(graph->channel,c*s2,0,1);//16
    out_shape.push_back(oc);
    lambda.push_back(Parser::make_affine({oc.name,"%",c})); // oc% 4
    auto oh = Graph::get_variable(graph->input_h,h/stride,0,3);//1
    out_shape.push_back(oh);
    lambda.push_back(Parser::make_affine({oh.name,"*",stride,"+",oc.name,"/",c,"/",stride}));// oh*stride+oc/c/stride
    auto ow = Graph::get_variable(graph->input_w,w/stride,0,4);//1
    out_shape.push_back(ow);
    lambda.push_back(Parser::make_affine({ow.name,"*",stride,"+",oc.name,"/",c,"%",stride}));//ow*stride+oc/c%stride

    auto t = new Tensor;
    graph->push(new Assign(t,_input,out_shape,graph->num_cpt,lambda));//autogen formula
    outputs[0] = t;
}