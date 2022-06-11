//
// Created by sun on 2022/1/3.
//
#include"layer.h"

TensorHandle batch_norm_infer(Graph* graph,TensorHandle input,float eps,const string& name){
    /*
     *  assuming data layout is NCHW
     */
    int c = input->shape[1].get_upper_bound();
    auto mean = graph->new_weight(c,1,DT_FLOAT,name+"_mean");
    auto var = graph->new_weight(c,1,DT_FLOAT,name+"_var");
    auto scale = graph->new_weight(c,1,DT_FLOAT,name+"_scale");
    auto bias = graph->new_weight(c,1,DT_FLOAT,name+"_bias");
    return graph->batch_norm(input, scale, bias, mean, var, eps);
}

TensorHandle conv2d(Graph* graph, TensorHandle input, int oc, int kernel, int stride, PaddingMode p, const string& name){
    int ic = input->shape[1].get_upper_bound();
    TensorHandle weight = graph->new_weight({oc,ic,kernel,kernel},DT_FLOAT,name+"_weight");
    return graph->conv2d(input,weight,stride,stride,p);
}

TensorHandle conv3d(Graph* graph, TensorHandle input, int oc, int kernel, int stride, PaddingMode p, const string& name){
    int ic = input->shape[1].get_upper_bound();
    TensorHandle weight = graph->new_weight({oc,ic,kernel,kernel,kernel},DT_FLOAT,name+"_weight");
    return graph->conv3d(input,weight,stride,stride,stride,p);
}

TensorHandle conv2d_transpose(Graph* graph, TensorHandle input, int oc, int kernel, int stride, const string& name,const vector<int>& out_padding){
    // OIHW
    int ic = input->shape[1].get_upper_bound();
    TensorHandle weight = graph->new_weight({oc,ic,kernel,kernel},DT_FLOAT,name+"_weight");
    return graph->conv2d_transpose(input,weight,stride,stride,out_padding);
}

TensorHandle dense_add_bias(Graph* graph,TensorHandle input,int units,const string&name){
    int k = input->shape.back().get_upper_bound();
    TensorHandle weight = graph->new_weight({k,units},DT_FLOAT,name+"_weight");
    TensorHandle bias = graph->new_weight(units,1,DT_FLOAT,name+"_bias"); // axis=1?
    TensorHandle dense = graph->dense(input,weight);
    return graph->bias_add(dense,bias);
}

TensorHandle bias_add(Graph* graph,TensorHandle input,const string&name){
    TensorHandle bias = graph->new_weight(input->shape[1].get_upper_bound(),1,DT_FLOAT,name+"_bias"); // axis=1?
    return graph->bias_add(input,bias);
}