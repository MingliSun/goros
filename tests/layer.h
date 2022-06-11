//
// Created by sun on 2021/12/27.
//

#ifndef AUTOFUSION_LAYER_H
#define AUTOFUSION_LAYER_H

#include"Ops.h"

TensorHandle batch_norm_infer(Graph* graph,TensorHandle input,float eps,const string& name);

TensorHandle conv2d(Graph* graph, TensorHandle input, int oc, int kernel, int stride, PaddingMode p, const string& name);

TensorHandle conv3d(Graph* graph, TensorHandle input, int oc, int kernel, int stride, PaddingMode p, const string& name);

TensorHandle conv2d_transpose(Graph* graph, TensorHandle input, int oc, int kernel, int stride, const string& name,const vector<int>& out_padding={0,0});

TensorHandle dense_add_bias(Graph* graph,TensorHandle input,int units,const string&name);

TensorHandle bias_add(Graph* graph,TensorHandle input,const string&name);

#endif //AUTOFUSION_LAYER_H
