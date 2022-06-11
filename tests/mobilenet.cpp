//
// Created by sun on 2022/1/3.
//

#include "layer.h"
TensorHandle conv_block(Graph* graph,TensorHandle input,const string& name,int oc,int kernel=3,int stride=1){
    auto conv = conv2d(graph,input,oc,kernel,stride,PD_MODE_SAME,name+"_conv");
    auto bn = batch_norm_infer(graph,conv,2e-5,name+"_bn");
    return  graph->relu(bn);
}

TensorHandle separable_conv_block(Graph* graph,TensorHandle input,const string& name,
                                  int depthwise_channels,
                                  int pointwise_channels,
                                  int stride=1,
                                  int kernel=3,
                                  float epsilon = 1e-5
                                  ){
    auto weight = graph->new_weight({depthwise_channels,1,kernel,kernel},DT_FLOAT,name+"_conv1_weight");
    auto conv1 = graph->conv2d_group(input,weight,stride,stride,PD_MODE_SAME);
    auto bn1 = batch_norm_infer(graph,conv1,epsilon,name+"_bn1");
    auto act1 = graph->relu(bn1);

    auto conv2 = conv2d(graph,act1,pointwise_channels,1,1,PD_MODE_SAME,name+"_conv2");
    auto bn2 = batch_norm_infer(graph,conv2,epsilon,name+"bn2");
    return graph->relu(bn2);
}

Graph* mobile_net(int num_classes=1000,
                  const vector<int>& data_shape={1, 3, 224, 224},
                  float alpha=1.0,
                  bool is_shallow=false){
    /*
     * Function to construct a MobileNet
     */
    auto graph = new Graph;
    auto data = graph->new_input(data_shape,DT_FLOAT,"data");
    auto body = conv_block(graph,data,"conv_bock_1",int(32*alpha),3,2);
    body = separable_conv_block(graph,body,"separable_conv_block_1",int(32*alpha),int(64*alpha));
    body = separable_conv_block(graph,body,"separable_conv_block_2",  int(64 * alpha),int(128 * alpha),2);
    body = separable_conv_block(graph,body,"separable_conv_block_3",  int(128 * alpha),int(128 * alpha));
    body = separable_conv_block(graph,body,"separable_conv_block_4",  int(128 * alpha),int(256 * alpha),2);
    body = separable_conv_block(graph,body,"separable_conv_block_5",  int(256 * alpha),int(256 * alpha));
    body = separable_conv_block(graph,body,"separable_conv_block_6",  int(256 * alpha),int(512 * alpha),2);
    if(is_shallow){
        body = separable_conv_block(graph,body,"separable_conv_block_7",  int(512 * alpha),int(1024 * alpha));
        body = separable_conv_block(graph,body,"separable_conv_block_8",  int(1024 * alpha),int(1024 * alpha),2);
    }else{
        for(int i=7;i<12;i++){
            body = separable_conv_block(graph,body,"separable_conv_block_"+::to_string(i),  int(512 * alpha),int(512 * alpha));
        }
        body = separable_conv_block(graph,body,"separable_conv_block_12",  int(512 * alpha),int(1024 * alpha),2);
        body = separable_conv_block(graph,body,"separable_conv_block_13",  int(1024 * alpha),int(1024 * alpha));
    }
    auto pool = graph->global_avg_pool2d(body);
    auto flatten = graph->batch_flatten(pool);
    auto fc = dense_add_bias(graph,flatten,num_classes,"fc");
    auto softmax = graph->softmax(fc);
    graph->function({softmax});
    return graph;
}

void test_mobile_net(){
    auto graph = mobile_net();
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}

void test_mobile_net_1(){
    const vector<int>& data_shape={1, 3, 224, 224};
    float alpha=1.0;
    auto graph = new Graph;
    auto data = graph->new_input(data_shape,DT_FLOAT,"data");
    auto body = conv_block(graph,data,"conv_bock_1",int(32*alpha),3,2);
    body = separable_conv_block(graph,body,"separable_conv_block_1",int(32*alpha),int(64*alpha));
    graph->function({body});

    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}