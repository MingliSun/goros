//
// Created by sun on 2022/1/3.
//
#include"layer.h"

static TensorHandle output38;
static TensorHandle output76;

TensorHandle mish(Graph* graph,TensorHandle input){
    auto body = graph->tanh(graph->log(input));
    return graph->multiply(input,body);
}

TensorHandle dbl(Graph* graph, TensorHandle input, int oc, int kernel, int stride, const string& name){
    PaddingMode p;
    if(stride==2) p = PD_MODE_VALID;
    else p = PD_MODE_SAME;
    auto body = conv2d(graph,input,oc,kernel,stride,p,name+"_conv");
    body = batch_norm_infer(graph,body,1e-4,name+"_bn");
    return graph->leaky_relu(body);
}

TensorHandle dbm(Graph* graph, TensorHandle input, int oc, int kernel, int stride, const string& name){
    PaddingMode p;
    if(stride==2) p = PD_MODE_VALID;
    else p = PD_MODE_SAME;
    auto body = conv2d(graph,input,oc,kernel,stride,p,name+"_conv");
    body = batch_norm_infer(graph,body,1e-4,name+"_bn");
    return mish(graph,body);
}

TensorHandle residual_unit(Graph* graph,TensorHandle input,int num_filters,int num_blocks,bool all_narrow = true){
    /*
     * A series of residual unit starting with a down sampling Convolution2D
     */
    auto pre_conv1 = graph->pad(input,{2,3},{1,0,1,0});
    pre_conv1 = dbm(graph,pre_conv1,num_filters,3,2,"pre_conv1");
    TensorHandle short_conv,main_conv;
    int oc;
    if(all_narrow) oc = num_filters/2;
    else oc = num_filters;
    short_conv = dbm(graph,pre_conv1,oc,1,1,"short_conv");
    main_conv = dbm(graph,pre_conv1,oc,1,1,"main_conv");

    for(int i=0;i<num_blocks;i++){
        auto y= dbm(graph,main_conv,num_filters/2,1,1,"");
        y = dbm(graph,y,oc,3,1,"");
        main_conv = graph->add(main_conv,y);
    }
    auto post_conv = dbm(graph,main_conv,oc,1,1,"");
    auto route = graph->concat({post_conv,short_conv},1);
    return dbm(graph,route,num_filters,1,1,"");
}

TensorHandle darknet53_body(Graph* graph,TensorHandle input){
    auto x = dbm(graph,input,32,3,1,"");
    x = residual_unit(graph,x,64,1,false);
    x = residual_unit(graph,x,128,2);
    x = residual_unit(graph,x,256,8);
    output76 = x;
    x = residual_unit(graph,x,512,8);
    output38 = x;
    x = residual_unit(graph,x,1024,4);
    //output19
    return x;
}

Graph* yolo_v4(int num_anchors,int num_classes){
    auto graph = new Graph;
    auto input = graph->new_input({1,3,608,608});
    auto darknet = darknet53_body(graph,input);
    ////19x19 head
    auto y19 = dbl(graph,darknet,512,1,1,"");
    y19 = dbl(graph,y19,1024,3,1,"");
    y19 = dbl(graph,y19,512,1,1,"");
    auto pool1 = graph->max_pool2d(y19,13,13,1,1,PD_MODE_SAME);
    auto pool2 = graph->max_pool2d(y19,9,9,1,1,PD_MODE_SAME);
    auto pool3 = graph->max_pool2d(y19,5,5,1,1,PD_MODE_SAME);
    y19 = graph->concat({pool1,pool2,pool3,y19},1);
    y19 = dbl(graph,y19,512,1,1,"");
    y19 = dbl(graph,y19,1024,3,1,"");
    y19 = dbl(graph,y19,512,1,1,"");
    auto y19_upsample = graph->upsampling(y19,2,2);
    y19_upsample = dbl(graph,y19_upsample,256,1,1,"");
    ////38x38 head
    auto y38 = dbl(graph,output38,256,1,1,"");
    y38 = graph->concat({y38,y19_upsample},1);
    y38 = dbl(graph,y38,256,1,1,"");
    y38 = dbl(graph,y38,512,3,1,"");
    y38 = dbl(graph,y38,256,1,1,"");
    y38 = dbl(graph,y38,512,3,1,"");
    y38 = dbl(graph,y38,256,1,1,"");
    auto y38_upsample = graph->upsampling(y38,2,2);
    y38_upsample = dbl(graph,y38_upsample,128,1,1,"");
    ////76x76 head
    auto y76 = dbl(graph,output76,128,1,1,"");
    y76 = graph->concat({y76,y38_upsample},1);
    y76  = dbl(graph,y76,128,1,1,"");
    y76  = dbl(graph,y76,256,3,1,"");
    y76  = dbl(graph,y76,128,1,1,"");
    y76  = dbl(graph,y76,256,3,1,"");
    y76  = dbl(graph,y76,128,1,1,"");
    ////76x76 output
    auto y76_output = dbl(graph,y76,256,3,1,"");
    y76_output = conv2d(graph,y76_output,num_anchors*(num_classes+5),1,1,PD_MODE_SAME,"");
    //// 38x38 output
    auto y76_downsample = graph->pad(y76,{2,3},{1,0,1,0});
    y76_downsample = dbl(graph,y76_downsample,256,3,2,"");
    y38 = graph->concat({y76_downsample,y38},1);
    y38 = dbl(graph,y38,256,1,1,"");
    y38 = dbl(graph,y38,512,3,1,"");
    y38 = dbl(graph,y38,256,1,1,"");
    y38 = dbl(graph,y38,512,3,1,"");
    y38 = dbl(graph,y38,256,1,1,"");
    auto y38_output = dbl(graph,y38,512,3,1,"");
    y38_output = conv2d(graph,y38_output,num_anchors*(num_classes+5),1,1,PD_MODE_SAME,"");
    ////19x19 output
    auto y38_downsample = graph->pad(y38,{2,3},{1,0,1,0});
    y38_downsample = dbl(graph,y38_downsample,512,3,2,"");
    y19 = graph->concat({y38_downsample,y19},1);
    y19 = dbl(graph,y19,512,1,1,"");
    y19 = dbl(graph,y19,1024,3,1,"");
    y19 = dbl(graph,y19,512,1,1,"");
    y19 = dbl(graph,y19,1024,3,1,"");
    y19 = dbl(graph,y19,512,1,1,"");
    auto y19_output = dbl(graph,y19,1024,3,1,"");
    y19_output = conv2d(graph,y19_output,num_anchors*(num_classes+5),1,1,PD_MODE_SAME,"");
    //todo after process: reshape split sigmoid concat reshape like yolo_v3 did

    graph->function({y19_output,y38_output,y76_output});

    return graph;
}