//
// Created by sun on 2021/11/18.
//
#include"Pattern.h"
#include"Parser.h"
#include<cassert>
#include<iostream>
#include<queue>
#include"Producer.h"


//template < typename  T>
//void *Producer::allocate_sequential(const vector<Variable>& shape) {
//    assert(!shape.empty());
//    int size = 1;
//    for(auto& v:shape){
//        size*= v.get_upper_bound();
//    }
//    T* result =  new T[size];
//    for(int i=0;i<size;i++){
//        result[i] = i+1;
//    }
//    return result;
//}

//template < typename  T>
//void *Producer::allocate_zero(const vector<Variable>& shape) {
//    assert(!shape.empty());
//    int size = 1;
//    for(auto& v:shape){
//        size*= v.get_upper_bound();
//    }
//    T* result = new T[size]{0};
//    return result;
//}



PatternMatch::PatternMatch(Graph *g) :graph(g){

}
//Heuristic should we modify WEIGHT into MIX ,more aggressive...... need to see more examples
bool PatternMatch::add2add_left(int idx) {
    //see if we can eliminate common code (delete if_else)
    if(graph->computations[idx]->op==ADD&& graph->computations[idx]->in[0]->sub_op_type == ADD){
        int left_idx = graph->computations[idx]->in[0]->index;
        auto top = dynamic_cast<Add*>(graph->computations[idx]);
        auto left = dynamic_cast<Add*>(graph->computations[left_idx]);
        if(is_multiple_usage(left)) return false;
        if(left->out->tensorType==MIX&&left->in[1]->tensorType==WEIGHT&&top->in[1]->tensorType==WEIGHT){
            release({top,left});
            auto second = new Add(left->out,left->in[1],top->in[1],left_idx);
            auto first  = new Add(top->out,left->in[0],left->out,idx);
            delete_t({top,left});
            graph->computations[idx] = first;
            graph->computations[left_idx] = second;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }else if(left->out->tensorType==MIX&&left->in[0]->tensorType==WEIGHT&&top->in[1]->tensorType==WEIGHT){
            release({top,left});
            auto second = new Add(left->out,left->in[0],top->in[1],left_idx);
            auto first  = new Add(top->out,left->in[1],left->out,idx);
            delete_t({top,left});
            graph->computations[idx] = first;
            graph->computations[left_idx] = second;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }
    return false;
}

bool PatternMatch::mul2mul_left(int idx) {
    if(graph->computations[idx]->op==MUL&& graph->computations[idx]->in[0]->sub_op_type == MUL){
        int left_idx = graph->computations[idx]->in[0]->index;
        auto top = dynamic_cast<Mul*>(graph->computations[idx]);
        auto left = dynamic_cast<Mul*>(graph->computations[left_idx]);
        if(is_multiple_usage(left)) return false;
        vector<Variable> simplify;
        if(!top->simplify_const_tensor_indices.empty()||!left->simplify_const_tensor_indices.empty()){
            simplify = top->simplify_const_tensor_indices;
            simplify.insert(simplify.end(),left->simplify_const_tensor_indices.begin(),left->simplify_const_tensor_indices.end());
        }
        if(left->out->tensorType==MIX&&top->in[1]->tensorType==WEIGHT&&
                (left->in[1]->tensorType!=INPUT||left->in[0]->tensorType!=INPUT)){
            release({top,left});
            Computation* first=nullptr,*second=nullptr;
            //priority: WEIGHT > MIX
            if(left->in[1]->tensorType==WEIGHT&&is_contain(left->in[1]->shape,top->in[1]->shape)){
                second = new Mul(left->out,left->in[1],top->in[1],left_idx);
                first  = new Mul(top->out,left->in[0],left->out,idx,simplify);
            }else if(left->in[0]->tensorType==WEIGHT&&is_contain(left->in[0]->shape,top->in[1]->shape)){
                second = new Mul(left->out,left->in[0],top->in[1],left_idx);
                first  = new Mul(top->out,left->in[1],left->out,idx,simplify);
            }else if(left->in[1]->tensorType==MIX&&is_contain(left->in[1]->shape,top->in[1]->shape)){// weight is not contain y
                second = new Mul(left->out,left->in[1],top->in[1],left_idx);
                first  = new Mul(top->out,left->out,left->in[0],idx,simplify);
            }else if(left->in[0]->tensorType==MIX&&is_contain(left->in[0]->shape,top->in[1]->shape)){
                second = new Mul(left->out,left->in[0],top->in[1],left_idx);
                first  = new Mul(top->out,left->out,left->in[1],idx,simplify);
            }else if(left->in[1]->tensorType==WEIGHT){// mix is not contained
                second = new Mul(left->out,left->in[1],top->in[1],left_idx);
                first  = new Mul(top->out,left->in[0],left->out,idx,simplify);
            }else if(left->in[0]->tensorType==WEIGHT){
                second = new Mul(left->out,left->in[0],top->in[1],left_idx);
                first  = new Mul(top->out,left->in[1],left->out,idx,simplify);
            }

            delete_t({top,left});
            graph->computations[idx] = first;
            graph->computations[left_idx] = second;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }
    return false;
}

void PatternMatch::optimize() {
    graph->frontIdx = graph->num_cpt-1;
    bool flag=true;
    int n=20;
    while(flag){
        int count=0;
        for(int i=graph->num_cpt-1;i>=0;i--){
            if(!graph->computations[i]) {count+=n; continue;}
            if(!mul2smax_left(i)) count++;
            else cout<<"mul2smax_left"<<endl;
            if(!add_sum_mul_assign_left(i)) count++;
            else cout<<"mul2smax_left"<<endl;
            if(!add2add_left(i)) count++;
            else cout<<"add2add"<<endl;
            if(!mul2mul_left(i)) count++;
            else cout<<"mul2mul"<<endl;
            if(!mul2add_left(i)) count++;
            else cout<<"mul2add"<<endl;
            if(!mul2sum_left(i)) count++;
            else cout<<"mul2sum"<<endl;
            if(!add2sum_both(i)) count++;
            else cout<<"add2sum"<<endl;
            if(!sum2mul_single(i)) count++;
            else cout<<"sum2mul"<<endl;
            if(!assign2assign_single(i)) count++;
            else cout<<"assign2assign"<<endl;
            if(!cond2mul_left(i)) count++;
            else cout<<"cond2mul_left"<<endl;
            if(!cond2mul_both(i)) count++;
            else cout<<"cond2mul"<<endl;
            if(!cond2assign_both(i)) count++;
            else cout<<"cond2assign"<<endl;
            if(!cond_sum_mul_assign_left(i)) count++;
            else cout<<"cond_sum_mul_assign"<<endl;
            if(!cond2sum_both(i)) count++;
            else cout<<"cond_sum_both"<<endl;
            if(!cond2smax_both(i)) count++;
            else cout<<"cond2smax_both"<<endl;
            if(!reverse_both(i)) count++;
            else cout<<"reverse_both"<<endl;
            if(!mul2assign_left(i)) count++;
            else cout<<"mul2assign_left"<<endl;
            if(!mul2assign_both(i)) count++;
            else cout<<"mul2assign_both"<<endl;
            if(!add2assign_both(i)) count++;
            else cout<<"add2assign_both"<<endl;
            if(!reshape2assign_single(i)) count++;
            else cout<<"reshape2assign_single"<<endl;
        }
        if(count==graph->num_cpt*n) flag = false;
        cout<<"cycle!"<<endl;
    }
    for(auto&p:subst_history){
        cout<<p.first<<" at "<<p.second<<endl;
    }
    //preprocess_weights();
    cout<<"preprocess_weights success or shut down!"<<endl;
    partition();
    cout<<"partition success!"<<endl;
}

bool PatternMatch::mul2add_left(int idx) {
    if(graph->computations[idx]->op==MUL&& graph->computations[idx]->in[0]->sub_op_type == ADD){
        int left_idx = graph->computations[idx]->in[0]->index;
        auto top = dynamic_cast<Mul*>(graph->computations[idx]);
        auto left = dynamic_cast<Add*>(graph->computations[left_idx]);
        if(is_multiple_usage(left)) return false;
        if(left->out->tensorType==MIX&&top->in[1]->tensorType==WEIGHT&&(left->in[0]->tensorType==WEIGHT||left->in[1]->tensorType==WEIGHT)){
            release({top,left});
            auto t = new Tensor,t1 = new Tensor;
            auto down_one = new Mul(t,left->in[0],top->in[1],left_idx);
            auto down_two = new Mul(t1,left->in[1],top->in[1],graph->num_cpt);
            auto top_new = new Add(top->out,t,t1,idx);
            for(auto &p:top->out->related_idx){
                graph->computations[p.first]->in[p.second]->sub_op_type = ADD;
            }
            delete_t({top,left});
            graph->computations[idx] = top_new;
            graph->computations[left_idx] = down_one;
            graph->push(down_two);
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }
    return false;
}

bool PatternMatch::mul2sum_left(int idx) {
    if(graph->computations[idx]->op==MUL&& graph->computations[idx]->in[0]->sub_op_type == SUM) {
        int left_idx = graph->computations[idx]->in[0]->index;
        auto top = dynamic_cast<Mul*>(graph->computations[idx]);
        auto left = dynamic_cast<Sum*>(graph->computations[left_idx]);
        if(is_multiple_usage(left)) return false;
        // if top->in[1] has c ,it means oc
        vector<Variable> shape = shape_replace_with_variable(top->in[1]->shape,left->subst.first,left->subst.second);
        //// append constraint : is_dim_expand ,is that reasonable?
        if(!is_relevant(left->reduce_axis,shape)&&left->out->tensorType==MIX&&top->in[1]->tensorType==WEIGHT
                &&!is_dim_expand(left->in[0]->shape,shape)){
            release({top,left});
            auto t = new Tensor;
            top->in[1]->reshape(shape);//reshape
            auto second = new Mul(t,left->in[0],top->in[1],left_idx);
            auto first = new Sum(top->out,t,left->reduce_axis,idx,left->stride,left->is_boundary);
            for(auto &p:top->out->related_idx){
                graph->computations[p.first]->in[p.second]->sub_op_type = SUM;
            }
            graph->adjust_sum(first);
            delete_t({top,left});
            graph->computations[idx] = first;
            graph->computations[left_idx] = second;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }

    return false;
}

bool PatternMatch::add2sum_both(int idx) {
    if(graph->computations[idx]->op==ADD && graph->computations[idx]->in[0]->sub_op_type == SUM && graph->computations[idx]->in[1]->sub_op_type == SUM) {
        int left_idx = graph->computations[idx]->in[0]->index;
        int right_idx =graph->computations[idx]->in[1]->index;
        auto top = dynamic_cast<Add*>(graph->computations[idx]);
        auto down_left = dynamic_cast<Sum*>(graph->computations[left_idx]);
        auto down_right = dynamic_cast<Sum*>(graph->computations[right_idx]);
        //if add is not a elementwise-add ===> return false
        if(is_multiple_usage(down_left) || is_multiple_usage(down_right)||top->in[0]->shape!=top->in[1]->shape) return false;
        // if stride are not same, do not combine two same and do not enlarge kernel
        if(down_left->stride!=down_right->stride) return false;
        Variable a,b;
        Compare res = graph->is_single_concat_axis(down_left->reduce_axis,down_right->reduce_axis,a,b);
        if(res==COMPARE_EQ){
            int value = a.get_upper_bound()+b.get_upper_bound();
            Variable new_axis =  Graph::get_variable(graph->channel, value, 0, 1);//c
            vector<Variable> reduce_axis = shape_replace_with_variable(down_left->reduce_axis,a,new_axis);
            release({top,down_left,down_right});
            auto new_down = new Cond(down_left->out,down_left->in[0],down_right->in[0],a,b,new_axis,left_idx);
            auto new_top = new Sum(top->out,down_left->out,reduce_axis,idx,down_left->stride,down_left->is_boundary|down_right->is_boundary);
            for(auto &p:top->out->related_idx){
                graph->computations[p.first]->in[p.second]->sub_op_type = SUM;
            }
            delete_t({top,down_left,down_right});
            //todo we can record right_idx so we can insert a new computation at right_idx

            graph->adjust_sum(new_top);
            graph->computations[idx] = new_top;
            graph->computations[left_idx] = new_down;
            graph->computations[right_idx] = nullptr;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }else if(res==COMPARE_LT){
            down_left->pad_axis.resize(0);
            down_left->pads.resize(0);
            down_left->old_axis.resize(0);
            for(int i=0;i<down_left->reduce_axis.size();i++){
                if(down_left->reduce_axis[i]!=a){
                    down_left->old_axis.push_back(down_left->reduce_axis[i]);
                    down_left->pad_axis.push_back(down_right->reduce_axis[i]);
                    int value = (down_right->reduce_axis[i].get_upper_bound() - down_left->reduce_axis[i].get_upper_bound()+1)/2;//if pad%2!=0  pad at left and top
                    down_left->pads.push_back(value);
                }
            }
        }else if(res==COMPARE_GT){
            down_right->pad_axis.resize(0);
            down_right->pads.resize(0);
            down_right->old_axis.resize(0);
            for(int i=0;i<down_right->reduce_axis.size();i++){
                if(down_right->reduce_axis[i]!=b){
                    down_right->old_axis.push_back(down_right->reduce_axis[i]);
                    down_right->pad_axis.push_back(down_left->reduce_axis[i]);
                    int value = (down_left->reduce_axis[i].get_upper_bound() - down_right->reduce_axis[i].get_upper_bound()+1)/2;//if pad%2!=0  pad at left and top
                    down_right->pads.push_back(value);
                }
            }
        }
    }
    return false;
}

vector<Variable> PatternMatch::shape_replace_with_variable(const vector<Variable> &shape, const Variable& src, const Variable& tgt) {
    vector<Variable> result;
    for(const auto & i : shape){
        if(i!=src) result.push_back(i);
        else result.push_back(tgt);
    }
    return result;
}

bool PatternMatch::sum2mul_single(int idx) {
    if(graph->computations[idx]->op==SUM&& graph->computations[idx]->in[0]->sub_op_type == MUL) {
        int down_idx = graph->computations[idx]->in[0]->index;
        auto top = dynamic_cast<Sum*>(graph->computations[idx]);
        auto down = dynamic_cast<Mul*>(graph->computations[down_idx]);
        if(is_multiple_usage(down)) return false;
        if(!is_relevant(top->reduce_axis,down->in[1]->shape)&&down->in[1]->tensorType==WEIGHT){//assuming  in[0] is relevant to sum reduce_axis
            //todo
            //subst_history.emplace_back(__FUNCTION__ ,idx);
            return false;
        }else if(!top->pad_axis.empty()&&!top->pads.empty()&&!top->old_axis.empty()){
            ////Assuming we can pad channel either
            auto t = new Tensor,t1 = new Tensor;
            vector<Variable> out_shape = shape_replace_with_variable(down->in[0]->shape,top->old_axis,top->pad_axis);
            vector<Variable> out_shape2= shape_replace_with_variable(down->in[1]->shape,top->old_axis,top->pad_axis);

            map<Variable,Affine> mapping1;
            for(int i=0;i<top->pad_axis.size();i++){
                mapping1[top->old_axis[i]] = Parser::make_affine({top->pad_axis[i].name,"-",top->pads[i]});
            }
            release({top,down});
            graph->push(new Assign(t,down->in[0],out_shape,graph->num_cpt,mapping1));
            graph->push(new Assign(t1,down->in[1],out_shape2,graph->num_cpt,mapping1));
            auto new_down = new Mul(down->out,t,t1,down_idx);
            vector<Variable> reduce_axis = shape_replace_with_variable(top->reduce_axis,top->old_axis,top->pad_axis);
            auto new_top = new Sum(top->out,down->out,reduce_axis,idx,top->stride);
            delete_t({top,down});
            // don't know subst shape is useful or not
            graph->adjust_sum(new_top);

            graph->computations[idx] = new_top;
            graph->computations[down_idx] = new_down;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }
    return false;
}

bool PatternMatch::is_relevant(const vector<Variable>& reduce_axis, const vector<Variable>& shape) {
    int n = reduce_axis.size();
    int m = shape.size();
    assert(n>0);
    if(m==0) return false;
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            if(reduce_axis[i]==shape[j]) return true;
        }
    }
    return false;
}

vector<Variable> PatternMatch::shape_replace_with_variable(const vector<Variable> &shape, const vector<Variable> &src,
                                                           const vector<Variable> &tgt) {
    assert(src.size()==tgt.size());
    vector<Variable> result;
    for(const auto & i : shape){
        bool found = false;
        int j=0;
        for(;j<src.size();j++){
            if(i==src[j]){
                found = true;
                break;
            }
        }
        if(found) result.push_back(tgt[j]);
        else result.push_back(i);
    }
    return result;
}

//void PatternMatch::adjust_sum(Sum* sum) {
//    if(graph->weight_ic.find(sum->out->shape[1].get_upper_bound())!=graph->weight_ic.end()) {
//        auto v = graph->weight_ic[sum->out->shape[1].get_upper_bound()];
//        sum->subst = make_pair(v,sum->out->shape[1]);
//        sum->out->shape[1] = v;
//    }else{
//        graph->weight_ic[sum->out->shape[1].get_upper_bound()] = sum->out->shape[1];
//    }
//}

bool PatternMatch::assign2assign_single(int idx) {
    if(graph->computations[idx]->op==ASSIGN&& graph->computations[idx]->in[0]->sub_op_type == ASSIGN){
        int down_idx = graph->computations[idx]->in[0]->index;
        auto top = dynamic_cast<Assign*>(graph->computations[idx]);
        auto down = dynamic_cast<Assign*>(graph->computations[down_idx]);
        for(auto p:down->out->related_idx){
            if(graph->computations[p.first]->op!=ASSIGN) return false; // all usage is assign
        }
//        if(is_multiple_usage(down)) return false;
        if(top->out->shape==down->in[0]->shape){
            release({top,down});
            for(auto &p:top->out->related_idx){
                // todo that could be memory leak==>need to fix it
                graph->computations[p.first]->in[p.second] = down->in[0];
                down->in[0]->related_idx[p.first] = p.second;
            }
            delete_t({top,down});
            graph->computations[idx] = nullptr;
            graph->computations[down_idx] = nullptr;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }else {
            // todo it is  o^2 complexity ,see if we can simplify the algorithm
            //bug report: if top->mapping has dependent item , like a->b  b->c , we must replace b->c first,sorting can be help
            //if we have around a->b b->c c->a ,it can not be resolved yet,leave it a todo: we can rename it with different variable: a->w1 b->w2 c->w3 and w1->b w2->c w3->a
            //for now, we only do sorting
            // do not change down,we can change top
            vector<pair<Variable,Affine>> bucket;
            vector<Affine> down_lambda_cp = down->lambda;
            for(auto&a:down_lambda_cp){ //Affine
                for(auto&m:top->mapping){//Straight Insertion Sort
                    bool inserted = false;
                    for(int i=0;i<bucket.size();i++){
                        if(bucket[i].second.variables.find(m.first.name)!=bucket[i].second.variables.end()){
                            bucket.insert(bucket.begin()+i,make_pair(m.first,m.second));
                            inserted = true;
                            break;
                        }
                    }
                    if(!inserted) bucket.emplace_back(m.first,m.second);
                }
                for(auto& p:bucket){
                    a.replace(p.first.name,p.second);
                }
//                cout<<a.to_string()<<endl;
            }
            if(down->formula){
                for(auto& p:bucket){
                    down->formula->replace(p.first.name,p.second);
                }
            }
            Formula* formula ;
            if(down->formula){
                formula = down->formula->conjunction(top->formula);
            }
            else formula = top->formula;
            // down->formula changes as down->lambda because they share same memory ---fixed
            release({top}); // remove connection between top and down
            if(down->out->related_idx.empty()){
                release({down});
            }
            auto new_top = new Assign(top->out,down->in[0],top->out->shape,idx,down_lambda_cp,formula);
            delete_t({top});
            if(down->out->related_idx.empty()){
                delete_t({down});
            }
            graph->computations[idx] = new_top;
            if(down->out->related_idx.empty()){
                graph->computations[down_idx] = nullptr;
            }
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }
    return false;
}

bool PatternMatch::cond2mul_both(int idx) {
    if(graph->computations[idx]->op==COND && graph->computations[idx]->in[0]->sub_op_type == MUL && graph->computations[idx]->in[1]->sub_op_type == MUL){
        int left_idx = graph->computations[idx]->in[0]->index;
        int right_idx = graph->computations[idx]->in[1]->index;
        auto top = dynamic_cast<Cond*>(graph->computations[idx]);
        auto left = dynamic_cast<Mul*>(graph->computations[left_idx]);
        auto right = dynamic_cast<Mul*>(graph->computations[right_idx]);
        CondMode one = get_cond_mode(left->in[0],right->in[0],top->in0_axis,top->in1_axis);
        CondMode two = get_cond_mode(left->in[1],right->in[1],top->in0_axis,top->in1_axis);
        if(one==CondMode_INVALID ||two==CondMode_INVALID || is_multiple_usage(left) || is_multiple_usage(right)) return false;
        if(one==CondMode_VALID&&two==CondMode_VALID){
            release({top,left,right});
            auto down_left = new Cond(left->out,left->in[0],right->in[0],top->in0_axis,top->in1_axis,top->new_axis,left_idx);
            auto down_right = new Cond(right->out,left->in[1],right->in[1],top->in0_axis,top->in1_axis,top->new_axis,right_idx);
            auto new_top = new Mul(top->out,left->out,right->out,idx);
            for(auto &p:top->out->related_idx){
                graph->computations[p.first]->in[p.second]->sub_op_type = MUL;
            }
            delete_t({top,left,right});
            graph->computations[idx] = new_top;
            graph->computations[left_idx] = down_left;
            graph->computations[right_idx] = down_right;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }else if(one==CondMode_VALID&&two==CondMode_SAME){
            release({top,left,right});
            auto down_left = new Cond(left->out,left->in[0],right->in[0],top->in0_axis,top->in1_axis,top->new_axis,left_idx);
            // left->in[1] or right->[1]
            auto new_top = new Mul(top->out,left->out,left->in[1],idx);//using left,then release right
            if(left->in[1]!=right->in[1]) release_isolated_chain(right->in[1]);
            for(auto &p:top->out->related_idx){
                graph->computations[p.first]->in[p.second]->sub_op_type = MUL;
            }
            delete_t({top,left,right});
            graph->computations[idx] = new_top;
            graph->computations[left_idx] = down_left;
            graph->computations[right_idx] = nullptr;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }else if(one==CondMode_SAME&&two==CondMode_VALID){
            release({top,left,right});
            auto down_left = new Cond(right->out,left->in[1],right->in[1],top->in0_axis,top->in1_axis,top->new_axis,left_idx);
            auto new_top = new Mul(top->out,left->in[0],right->out,idx); //using left then release right
            if(left->in[0]!=right->in[0]) release_isolated_chain(right->in[0]);
            for(auto &p:top->out->related_idx){
                graph->computations[p.first]->in[p.second]->sub_op_type = MUL;
            }
            delete_t({top,left,right});
            graph->computations[idx] = new_top;
            graph->computations[left_idx] = down_left;
            graph->computations[right_idx] = nullptr;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }
    return false;
}

bool PatternMatch::is_same_tensor_recur(TensorHandle x, TensorHandle y) {
    /*
     * compare two tensor
     */
    if(x->tensorType!=y->tensorType) return false;
    if(x->sub_op_type != y->sub_op_type) return false;
    if(x->sub_op_type == DATA) {// TensorType is input or weight
        // 0D weight compare contents (make sure it has content)
        // 0D input return x==y
        // nD weight compare contents (make sure it has content)
        // nD input return x==y
        if(x->tensorType==INPUT) {  return x == y;  } // do not record this, cause it is not time consuming
        //for two initial 0D weight
        //for two initial nD weight
        if( x->is_same_initial_weight(*y)){// record this
            same_tensor[x] = y;
            same_tensor[y] = x;
            return true;
        }
        return false;
    }
    if(graph->computations[x->index]->op!=graph->computations[y->index]->op) return false;
    // if op of the two is same , num_in is same
    // for sum max  reduce-type need to make sure their reduce_axis is <compatible> (compare result is not COMPARE_ER)
    //for assign need to make sure that their lambda and formula are <compatible> (is same after padding)
    //for cond need to make sure that concatenate axis is <compatible>(same position)
    //Now we just assume that the two is equal ignoring what mentioned above,it is a todo
    bool flag = true;
    bool record = true;
    for(int i=0;i<graph->computations[x->index]->num_in;i++){
        // Ignoring DATA&&WEIGHT type tensor even their shape are not same cause we assuming that we can padding shape of one to equal another
        // and can  mix two weight into one using concatenate or add,todo: judge if we can padding and if we can mix two weight into one weight
        // that is that could be mistake if two tensor is equal,but we don't record the fault result in theory,so that is ok.
        auto temp = graph->computations[x->index]->in[i];
        auto temp2 = graph->computations[y->index]->in[i];
        if(temp->tensorType==WEIGHT&&temp->sub_op_type==DATA&&temp2->tensorType==WEIGHT&&temp2->sub_op_type==DATA){
            record = is_same_tensor_recur(temp,temp2); //compare their contents are equal or not
            continue;
        }
        flag &= is_same_tensor_recur(temp,temp2);
    }
    if(flag&&record){
        same_tensor[x] = y;
        same_tensor[y] = x;
    }
    return flag;
}

bool PatternMatch::is_same_tensor(TensorHandle x, TensorHandle y) {
    if(x==y) return true;
    //assuming only two is same
    if(same_tensor[x]==y) return true;
    return is_same_tensor_recur(x,y);

}

CondMode PatternMatch::get_cond_mode(TensorHandle x, TensorHandle y, const Variable& c0,const Variable& c1) {
    int x_include_c =-1,y_include_c = -2;
    vector<Variable> x_shape,y_shape;
    for(int i=0;i<x->shape.size();i++){
        if(x->shape[i]==c0) x_include_c = i;
        else x_shape.push_back(x->shape[i]);
    }
    for(int i=0;i<y->shape.size();i++){
        if(y->shape[i]==c1)y_include_c = i;
        else y_shape.push_back(y->shape[i]);
    }
    if(x_include_c==y_include_c&&x_shape==y_shape) return CondMode_VALID;
    if(x_include_c==-1&&y_include_c==-2&&is_same_tensor(x,y)) return CondMode_SAME;
    return CondMode_INVALID;
}

bool PatternMatch::cond2assign_both(int idx) {
    if(graph->computations[idx]->op==COND && graph->computations[idx]->in[0]->sub_op_type == ASSIGN && graph->computations[idx]->in[1]->sub_op_type == ASSIGN){
        int left_idx = graph->computations[idx]->in[0]->index;
        int right_idx = graph->computations[idx]->in[1]->index;
        auto top = dynamic_cast<Cond*>(graph->computations[idx]);
        auto down_left = dynamic_cast<Assign*>(graph->computations[left_idx]);
        auto down_right = dynamic_cast<Assign*>(graph->computations[right_idx]);
        if(is_multiple_usage(down_left) || is_multiple_usage(down_right)) return false;
        map<string,string> m;
        Variable in0_axis,in1_axis;
        for(int i=0;i<down_left->out->shape.size();i++){
            if(down_left->out->shape[i]==top->in0_axis) { //assuming one of shape equals in0_axis
                if(down_left->lambda[i]!=Affine(down_left->out->shape[i])) return false;
                in0_axis = down_left->in[0]->shape[i];
                break;
            }
        }
        for(int i=0;i<down_right->out->shape.size();i++){
            if(down_right->out->shape[i]==top->in1_axis){//assuming one of shape equals in1_axis
                if(down_right->lambda[i]!=Affine(down_right->out->shape[i])) return false;
                in1_axis = down_right->in[0]->shape[i];
                break;
            }
        }
        m[top->in0_axis.name] = top->in1_axis.name;
        if(is_same_affines(down_left->lambda, down_right->lambda, m)&&
            is_same_formula(down_left->formula,down_right->formula,m)){
            for(auto& l:down_left->lambda){
                l.replace(top->in0_axis.name,Affine(top->new_axis));
            }
            if(down_left->formula) down_left->formula->replace(top->in0_axis.name,Affine(top->new_axis));
            release({top,down_left,down_right});
            auto down = new Cond(down_left->out, down_left->in[0], down_right->in[0], in0_axis, in1_axis, top->new_axis, left_idx);
            auto new_top = new Assign(top->out, down_left->out, top->out->shape, idx, down_left->lambda,down_left->formula);
            for(auto &p:top->out->related_idx){
                graph->computations[p.first]->in[p.second]->sub_op_type = ASSIGN;
            }
            delete_t({top,down_left,down_right});
            graph->computations[idx] = new_top;
            graph->computations[left_idx] = down;
            graph->computations[right_idx] = nullptr;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }else if(is_same_tensor(down_left->in[0],down_right->in[0])&&top->out->shape==down_left->in[0]->shape){ //primary judgement
            /*
             *  this is concat to split pass
             */
            auto second = Parser::make_affine({top->in1_axis.name,"+",top->in0_axis.get_upper_bound()});
            if(
                    (!down_left->formula||down_left->formula->autogen)&&
               (!down_right->formula||down_right->formula->autogen)&&
                    down_left->mapping.size()==1&&down_right->mapping.size()==1&&
                down_left->mapping.begin()->first==top->new_axis&&down_right->mapping.begin()->first==top->new_axis&&
                    down_left->mapping.begin()->second==Affine(top->in0_axis)&&down_right->mapping.begin()->second==second){
                release({top,down_left,down_right});
                //create nop but not using Nop
                //rebuild connection
                for(auto &p:top->out->related_idx){
                    //todo that could be memory leak==>need to fix it
                    graph->computations[p.first]->in[p.second] = down_left->in[0];
                    down_left->in[0]->related_idx[p.first] = p.second;
                }
                delete_t({top,down_left,down_right});
                graph->computations[idx] = nullptr;
                graph->computations[left_idx] = nullptr;
                graph->computations[right_idx] = nullptr;
                subst_history.emplace_back(__FUNCTION__ ,idx);
                return true;
            }
        }
    }
    return false;
}

bool PatternMatch::cond_sum_mul_assign_left(int idx) {
    //only insert Computation,do not need to check usages
    //double check stride (at sum and at assign)
    if(graph->computations[idx]->op==COND && graph->computations[idx]->in[0]->sub_op_type == SUM &&
            graph->computations[idx]->in[0]->tensorType==MIX &&graph->computations[idx]->in[1]->tensorType!=WEIGHT){
        auto cond  = dynamic_cast<Cond*>(graph->computations[idx]);
        int sum_idx = graph->computations[idx]->in[0]->index;
        auto sum = dynamic_cast<Sum*>(graph->computations[sum_idx]);
        bool concat_oc;
        if(sum->subst.first==cond->in0_axis) concat_oc=true;
        else concat_oc = false; //concatenate n axis
        if(sum->is_stride_equals(1)&&sum->in[0]->sub_op_type == MUL){
            int mul_idx = graph->computations[sum_idx]->in[0]->index;
            auto mul = graph->computations[mul_idx];
            //judge if  mul->in[1]->shape includes cond->in0_axis
            bool weight_include_axis = false;
            for(auto& s:mul->in[1]->shape){
                if((concat_oc&&s==sum->subst.second)||
                        (!concat_oc&&s==cond->in0_axis)) {weight_include_axis=true;break;}
            }
            if(weight_include_axis&&mul->in[0]->sub_op_type == ASSIGN && mul->in[1]->tensorType == WEIGHT){
                int assign_idx = graph->computations[mul_idx]->in[0]->index;
                auto assign = dynamic_cast<Assign*>(graph->computations[assign_idx]);
                // aggressive (don not need sub_op_type are same)
                //assign->in[0]->sub_op_type == cond->in[1]->sub_op_type

                if(assign->is_stride_equals(1)&&
                        concat_oc&&is_same_tensor(assign->in[0],cond->in[1]) ){// only support concat oc,assuming mul->in[1] is not a identity convolution weight
                    auto x = assign->in[0];
                    vector<Variable> shape =  mul->in[1]->shape;
                    if(shape[0].get_upper_bound()!=shape[1].get_upper_bound())
                        shape[0] = Graph::get_variable(graph->weight_oc,shape[1].get_upper_bound(),1,0);
                    TensorHandle iconv ;
                    auto sdt = Tensor::convert_data_type(x->dataType);
                    switch (sdt) {

                        case SDT_FLOAT:
                            iconv = new Tensor(shape,WEIGHT,x->dataType,Producer::allocate_iconv<float>(shape),"Iconv");
                            break;
                        case SDT_INT:
                            iconv = new Tensor(shape,WEIGHT,x->dataType,Producer::allocate_iconv<int>(shape),"Iconv");
                            break;
                        case SDT_UINT:
                            iconv = new Tensor(shape,WEIGHT,x->dataType,Producer::allocate_iconv<uint>(shape),"Iconv");
                            break;
                        case SDT_INVALID:
                            assert(false);
                            break;
                    }
                    auto t = new Tensor,t1 = new Tensor,t2 = new Tensor;
                    graph->push(new Assign(t,cond->in[1],assign->out->shape,graph->num_cpt,assign->lambda));
                    graph->push(new Mul(t1,t,iconv,graph->num_cpt));
                    auto s = new Sum(t2,t1,sum->reduce_axis,graph->num_cpt);
                    graph->adjust_sum(s);
                    graph->push(s);
                    // cut connects between <cond> and  cond->in[1]
                    cond->in[1]->related_idx.erase(cond->out->index);
                    cond->in[1] = t2;
                    t2->related_idx[cond->out->index] = 1;
                    subst_history.emplace_back(__FUNCTION__ ,idx);
                    return true;
                }
            }
        }
    }
    return false;
}

bool PatternMatch::cond2sum_both(int idx) {

    if(graph->computations[idx]->op==COND && graph->computations[idx]->in[0]->sub_op_type == SUM && graph->computations[idx]->in[1]->sub_op_type == SUM){
        int left_idx = graph->computations[idx]->in[0]->index;
        int right_idx = graph->computations[idx]->in[1]->index;
        auto top = dynamic_cast<Cond*>(graph->computations[idx]);
        auto down_left = dynamic_cast<Sum*>(graph->computations[left_idx]);
        auto down_right = dynamic_cast<Sum*>(graph->computations[right_idx]);

        // reduce axis cannot contain concat axis
        //check compare and check usages and check stride
        Compare res = compare_2_axis(down_left->reduce_axis,down_right->reduce_axis);
        if(res == COMPARE_ER || is_multiple_usage(down_left) || is_multiple_usage(down_right)||down_left->stride!=down_right->stride) return false;
        else if(res==COMPARE_EQ){
            if(down_left->subst.first==top->in0_axis&&down_right->subst.first==top->in1_axis){//oc
                release({top,down_left,down_right});
                auto new_axis = Graph::get_variable(Graph::weight_oc,top->new_axis.get_upper_bound(),1,0);//oc
                auto new_down = new Cond(down_left->out,down_left->in[0],down_right->in[0],down_left->subst.second,down_right->subst.second,new_axis,left_idx);
                auto new_top = new Sum(top->out,down_left->out,down_left->reduce_axis,idx);
                graph->adjust_sum(new_top);
                delete_t({top,down_left,down_right});
                graph->computations[idx] = new_top;
                graph->computations[left_idx] = new_down;
                graph->computations[right_idx] = nullptr;
                subst_history.emplace_back(__FUNCTION__ ,idx);
                return true;
            }else{//n h w
                release({top,down_left,down_right});
                auto new_down = new Cond(down_left->out,down_left->in[0],down_right->in[0],top->in0_axis,top->in1_axis,top->new_axis,left_idx);
                auto new_top = new Sum(top->out,down_left->out,down_left->reduce_axis,idx);
                graph->adjust_sum(new_top);
                delete_t({top,down_left,down_right});
                graph->computations[idx] = new_top;
                graph->computations[left_idx] = new_down;
                graph->computations[right_idx] = nullptr;
                subst_history.emplace_back(__FUNCTION__ ,idx);
                return true;
            }
        }else if(res==COMPARE_GT){
            down_right->pad_axis.resize(0);
            down_right->pads.resize(0);
            down_right->old_axis.resize(0);
            for(int i=0;i<down_right->reduce_axis.size();i++){
                if(down_right->reduce_axis[i]!=down_left->reduce_axis[i]){
                    down_right->old_axis.push_back(down_right->reduce_axis[i]);
                    down_right->pad_axis.push_back(down_left->reduce_axis[i]);
                    int value = (down_left->reduce_axis[i].get_upper_bound() - down_right->reduce_axis[i].get_upper_bound()+1)/2;//if pad%2!=0  pad at left and top
                    down_right->pads.push_back(value);
                }
            }
        }else if(res==COMPARE_LT){
            down_left->pad_axis.resize(0);
            down_left->pads.resize(0);
            down_left->old_axis.resize(0);
            for(int i=0;i<down_left->reduce_axis.size();i++){
                if(down_left->reduce_axis[i]!=down_right->reduce_axis[i]){
                    down_left->old_axis.push_back(down_left->reduce_axis[i]);
                    down_left->pad_axis.push_back(down_right->reduce_axis[i]);
                    int value = (down_right->reduce_axis[i].get_upper_bound() - down_left->reduce_axis[i].get_upper_bound()+1)/2;//if pad%2!=0  pad at left and top
                    down_left->pads.push_back(value);
                }
            }
        }
    }
    return false;
}

void PatternMatch::preprocess_weights() {
    for(int i=graph->num_cpt-1;i>=0;i--) {
        if (!graph->computations[i]) {
            continue;
        }
        if(graph->computations[i]->out->tensorType==WEIGHT&&!graph->computations[i]->out->data_ptr){
            preprocess_weights_recur(graph->computations[i]);
        }
    }
}

void PatternMatch::preprocess_weights_recur(Computation* _computation) {
    assert(_computation);
    for(int i=0;i<_computation->num_in;i++){
        if(!_computation->in[i]->data_ptr) {
            int index = _computation->in[i]->index;
            preprocess_weights_recur(graph->computations[index]);
        }
    }
    _computation->compute();
}

void PatternMatch::partition() const {
    int group_idx = 0;
    queue<Computation*> q;
    if(graph->outputs.empty()){
        auto c = graph->computations[graph->frontIdx];
        c->out->group_id.insert(group_idx);
        q.push(c);
        if(is_boundary(c)) group_idx++;
    }else{
        //default different group_idx for multiple output (e.g. yolo)
        //if want the same group_idx ,todo
        Computation* last;
        for(auto& c:graph->outputs){
            c->group_id.insert(group_idx++);
            last = graph->computations[c->index];
            q.push(last);
        }
        if(!is_boundary(last)) group_idx--;
    }
    vector<Computation*> candidate;// candidate of doing fine tune
    //bfs
    while(!q.empty()){
        auto computation = q.front();
        for(int i=0;i<computation->num_in;i++){
            if(computation->in[i]->tensorType!=WEIGHT&&computation->in[i]->sub_op_type!=DATA){ //-1
                auto temp = graph->computations[computation->in[i]->index];
                if(temp->out->related_idx.size()==1){//1
                    partition(temp,computation,group_idx);
                    q.push(temp);
                }else{ //>1
                    bool same = true;
                    bool parent_all_set = true;
                    auto iter = temp->out->related_idx.begin(); //map
                    auto& common = graph->computations[(iter)->first]->out->group_id; // set
                    for(auto j =iter;j!=temp->out->related_idx.end();j++){
                        if(graph->computations[j->first]->out->group_id.empty()) {parent_all_set = false;break;}
                        if(common!=graph->computations[j->first]->out->group_id) {same=false;break;}
                    }
                    if( parent_all_set&&same&& temp->out->group_id.empty()){
                        partition(temp,computation,group_idx);
                        q.push(temp);
                    }
                    if(parent_all_set&&!same&&temp->out->group_id.empty()){// keep record of temp and do fine tune
                        candidate.push_back(temp);
                        partition(temp,computation,group_idx,true);
                        q.push(temp);
                    }
                }
            }
        }
        q.pop();
    }
    //fine tune
    // merge single sub-operator to different neighborhood and do multiple(superfluous) computation
    int max_operator_num=1,max_num_consumer=2; // hyper-parameters
    for(auto k=candidate.begin();k!=candidate.end();){
        auto c = *k;
        vector<Computation*> temp;
        for(int i=0;i<graph->num_cpt;i++){
            if(graph->computations[i] && graph->computations[i]->out->group_id==c->out->group_id){
                temp.push_back(graph->computations[i]);
            }
        }
        if(temp.size()<=max_operator_num && c->out->related_idx.size()<=max_num_consumer){
            c->out->group_id.clear();
            for(auto p:c->out->related_idx){
                c->out->group_id.insert(graph->computations[p.first]->out->group_id.begin(),graph->computations[p.first]->out->group_id.end());
            }
            k++;
        }else{
            candidate.erase(k);
        }
    }
    //special case - split (fuse all  assign to its producer)
    for(int i=0;i<graph->num_cpt;i++) {
        auto tmp = graph->computations[i];
        if(tmp&&is_boundary(tmp)){
            // fuse all assign to its producer
            bool only_assign=true;
            auto iter = tmp->out->related_idx.begin();
            for(;iter!=tmp->out->related_idx.end();){
                auto consumer = graph->computations[(iter++)->first];
                if(consumer->op!=ASSIGN) {only_assign = false;break;} // tmp has other consumer than split ,then do not do fine tune
            }
            if(!only_assign) break;
            iter = tmp->out->related_idx.begin();
            for(;iter!=tmp->out->related_idx.end();){
                auto consumer = graph->computations[(iter++)->first];
                consumer->out->group_id = tmp->out->group_id;
                dynamic_cast<Assign*>(consumer)->is_boundary = true;
            }
            tmp->is_boundary = false; //FIXME - done
        }
    }

    //initialize fused operator
    vector<FusedOperator> bucket;
    graph->fused_operators.resize(group_idx);
    bucket.resize(group_idx);
    for(int i=0;i<group_idx;i++){
        graph->fused_operators[i].id = i;
    }

    // do partition
    for(int i=0;i<graph->num_cpt;i++){
        auto tmp = graph->computations[i];
        if(tmp&&!tmp->out->group_id.empty()){
            for(auto iter = tmp->out->group_id.begin();iter!=tmp->out->group_id.end();++iter){ //usually group_id has one element
                bucket[*iter].push(tmp);
            }
        }
    }
    //topological sorting
    for(int i=0;i<group_idx;i++){
        for(auto& computation:bucket[i].computations){
            bool found = false;
            for(auto c:candidate){
                if(c==computation){found=true;break;}
            }
            if(!found) init_num_consumer(computation);
            else  computation->num_consumer = 1;//see that computation has one consumer
        }
        while(!bucket[i].computations.empty()){
            for(auto j=bucket[i].computations.begin();j!=bucket[i].computations.end();){
                auto computation = *j;
               if(computation->num_consumer==0) {
                   graph->fused_operators[i].computations.push_back(computation);
                   bucket[i].computations.erase(j);
                   if(computation->in[0]&&computation->in[0]->index!=-1) graph->computations[computation->in[0]->index]->num_consumer--;
                   if(computation->in[1]&&computation->in[1]->index!=-1) graph->computations[computation->in[1]->index]->num_consumer--;
               }else{
                   j++;
               }
            }
        }
        reverse(graph->fused_operators[i].computations.begin(),graph->fused_operators[i].computations.end());
    }
    //done
}


bool PatternMatch::add_sum_mul_assign_left(int idx) {
    //Assuming graph->concat_axis == c
    //double check Iconv is qualified (at sum and at assign)
    if(graph->computations[idx]->op==ADD && graph->computations[idx]->in[0]->sub_op_type == SUM &&
       graph->computations[idx]->in[0]->tensorType==MIX &&graph->computations[idx]->in[1]->tensorType!=WEIGHT){
        auto add  = dynamic_cast<Add*>(graph->computations[idx]);
        int sum_idx = graph->computations[idx]->in[0]->index;
        auto sum = dynamic_cast<Sum*>(graph->computations[sum_idx]);
        if(add->in[0]->shape==add->in[1]->shape&&sum->in[0]->sub_op_type == MUL&&sum->is_stride_equals(1)){//sum stride=1 then add a identity convolution
            int mul_idx = graph->computations[sum_idx]->in[0]->index;
            auto mul = graph->computations[mul_idx];
            //Assuming mul->in[1] has concat axis
            if(mul->in[0]->sub_op_type == ASSIGN && mul->in[1]->tensorType == WEIGHT){
                int assign_idx = graph->computations[mul_idx]->in[0]->index;
                auto assign = dynamic_cast<Assign*>(graph->computations[assign_idx]);
                // do we need sub_op_type are same?
                // todo we do not need sub_op_type are same , it's more aggressive, it can erase a elementwise-add,don't know if it can improve efficiency
                // we can add a optimization option or optimization level
                if(assign->is_stride_equals(1)&&
                        mul->in[1]->shape[0].get_upper_bound()==add->in[1]->shape[1].get_upper_bound()&&// if add is elementwise-add,it satisfy
                    add->in[1]->sub_op_type==assign->in[0]->sub_op_type){//do we need non-linear constraint? todo
                    auto x = assign->in[0];
                    vector<Variable> shape =  mul->in[1]->shape;
                    vector<Variable> out_shape = assign->out->shape;
                    vector<Affine> new_lambda = assign->lambda; //since assign->is_stride_equals 1,we can use this lambda, if not ,we need to rebuild lambda
                    vector<Variable> new_axis = sum->reduce_axis;
                    if(shape[0].get_upper_bound()!=shape[1].get_upper_bound()){
                        shape[1] = Graph::get_variable(graph->channel, shape[0].get_upper_bound(), 0, 1);
                        out_shape[1] = shape[1];
                        new_lambda[1] = Affine(shape[1]);
                        //assuming c is at position 0 in reduce_axis,if not we need to replace c with shape[1] in right position(by traversal ),leave it a todo
                        new_axis[0] = shape[1];
                    }

                    //append shape[1] to graph->concat_axis
                    graph->concat_axis.insert(shape[1]);
                    TensorHandle iconv;
                    auto sdt = Tensor::convert_data_type(x->dataType);
                    switch (sdt) {

                        case SDT_FLOAT:
                            iconv = new Tensor(shape,WEIGHT,x->dataType,Producer::allocate_iconv<float>(shape),"Iconv");
                            break;
                        case SDT_INT:
                            iconv = new Tensor(shape,WEIGHT,x->dataType,Producer::allocate_iconv<int>(shape),"Iconv");
                            break;
                        case SDT_UINT:
                            iconv = new Tensor(shape,WEIGHT,x->dataType,Producer::allocate_iconv<uint>(shape),"Iconv");
                            break;
                        case SDT_INVALID:
                            assert(false);
                            break;
                    }
                    auto t = new Tensor,t1 = new Tensor,t2 = new Tensor;

                    graph->push(new Assign(t,add->in[1],out_shape,graph->num_cpt,new_lambda));
                    graph->push(new Mul(t1,t,iconv,graph->num_cpt));
                    auto s = new Sum(t2,t1,new_axis,graph->num_cpt);
                    graph->adjust_sum(s);
                    graph->push(s);
                    // cut connects between <add> and < ... >
                    add->in[1]->related_idx.erase(add->out->index);
                    add->in[1] = t2;
                    t2->related_idx[add->out->index]  = 1;
                    subst_history.emplace_back(__FUNCTION__ ,idx);
                    return true;
                }
            }
        }
    }
    return false;
}

bool PatternMatch::cond2smax_both(int idx) {
    if(graph->computations[idx]->op==COND && graph->computations[idx]->in[0]->sub_op_type == SMAX && graph->computations[idx]->in[1]->sub_op_type == SMAX){
        auto top = dynamic_cast<Cond*>(graph->computations[idx]);
        int down_left_index = graph->computations[idx]->in[0]->index;
        int down_right_index = graph->computations[idx]->in[1]->index;
        auto down_left = dynamic_cast<SMax*>(graph->computations[down_left_index]);
        auto down_right = dynamic_cast<SMax*>(graph->computations[down_right_index]);
        if(is_multiple_usage(down_left) || is_multiple_usage(down_right)) return false;
        if(is_same_tensor(down_left->in[1],down_right->in[1])){
            //Assuming  two inputs in each smax share same axis or right input is a scalar
            release({top,down_left,down_right});
            auto new_down = new Cond(down_left->out,down_left->in[0],down_right->in[0],top->in0_axis,top->in1_axis,top->new_axis,down_left_index);
            auto new_top = new SMax(top->out,down_left->out,down_left->in[1],idx);
            delete_t({top,down_right,down_left});
            graph->computations[idx] = new_top;
            graph->computations[down_left_index] = new_down;
            graph->computations[down_right_index] = nullptr;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }
    return false;
}

bool PatternMatch::cond2mul_left(int idx) {
    if(graph->computations[idx]->op==COND&& graph->computations[idx]->in[0]->sub_op_type == MUL){
        auto top = dynamic_cast<Cond*>(graph->computations[idx]);
        int left_idx = graph->computations[idx]->in[0]->index;
        auto left = dynamic_cast<Mul*>(graph->computations[left_idx]);
        // this function is prepared for cond2mul_both so we check_usages here
        if(is_multiple_usage(left)) return false;
        bool mul_right_include_axis = false;
        for(auto&p:left->in[1]->shape){
            if(p==top->in1_axis){mul_right_include_axis = true;break;}
        }
        //FIXME conditions are are wrong
        if((!mul_right_include_axis&&is_same_tensor(left->in[0],top->in[1]))||
            (mul_right_include_axis && left->in[0]->tensorType!=WEIGHT &&get_cond_mode(left->in[0], top->in[1], top->in0_axis,top->in1_axis) != CondMode_INVALID)||
            (mul_right_include_axis && left->in[1]->tensorType!=WEIGHT &&get_cond_mode(left->in[1], top->in[1], top->in0_axis,top->in1_axis) != CondMode_INVALID)){
            auto &in1 = left->in[1];
            auto one = new Tensor(in1->shape,WEIGHT,in1->dataType,Producer::allocate<float>(in1->shape,1.0),"one");

            if(left->in[0]->tensorType!=WEIGHT&&left->in[1]->tensorType==WEIGHT){
                auto t = new Tensor;
                //release connection
                top->in[1]->related_idx.erase(idx);
                //rebuild connection
                t->related_idx.insert(make_pair(idx,1));
                top->in[1] = t;
                graph->push(new Mul(t,top->in[1],one,graph->num_cpt));
                subst_history.emplace_back(__FUNCTION__ ,idx);
                return true;
            }else if(left->in[0]->tensorType==WEIGHT&&left->in[1]->tensorType!=WEIGHT){
                auto t = new Tensor;
                //release connection
                top->in[1]->related_idx.erase(idx);
                //rebuild connection
                t->related_idx.insert(make_pair(idx,1));
                top->in[1] = t;
                graph->push(new Mul(t,one,top->in[1],graph->num_cpt));

                subst_history.emplace_back(__FUNCTION__ ,idx);
                return true;
            }
        }
    }
    return false;
}

bool PatternMatch::mul2smax_left(int idx) {
    if(graph->computations[idx]->op==MUL&& graph->computations[idx]->in[0]->sub_op_type == SMAX){
        auto top = dynamic_cast<Mul*>(graph->computations[idx]);
        int left_idx = graph->computations[idx]->in[0]->index;
        auto left = dynamic_cast<SMax*>(graph->computations[left_idx]);
        if(is_multiple_usage(left)) return false;
        if(left->out->tensorType==MIX&&top->in[1]->tensorType==WEIGHT&&
                !is_dim_expand(left->out->shape,top->out->shape)&&
            (left->in[0]->tensorType==WEIGHT||left->in[1]->tensorType==WEIGHT)){
            release({top,left});
            auto t = new Tensor,t1 = new Tensor;
            auto down_one = new Mul(t,left->in[0],top->in[1],left_idx);
            auto down_two = new Mul(t1,left->in[1],top->in[1],graph->num_cpt);
            auto top_new = new SMax(top->out,t,t1,idx);
            for(auto &p:top->out->related_idx){
                graph->computations[p.first]->in[p.second]->sub_op_type = SMAX;
            }
            delete_t({top,left});
            graph->computations[idx] = top_new;
            graph->computations[left_idx] = down_one;
            graph->push(down_two);
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }
    return false;
}

Compare PatternMatch::compare_2_axis(const vector<Variable>&one,const vector<Variable>& two) {
    if(one.size()!=two.size()) return COMPARE_ER;
    int n = one.size();
    bool gt=true,lt=true,eq=true;
    for(int i=0;i<n;i++){
        if(one[i]==two[i]) lt &=true, gt&=true ,eq&=true;
        else if(one[i].get_upper_bound()>two[i].get_upper_bound()) gt&=true,lt&=false,eq&=false;
        else if(one[i].get_upper_bound()<two[i].get_upper_bound()) gt&=false,lt&=true,eq&=false;
    }
    if(eq) return COMPARE_EQ;
    else if(lt) return COMPARE_LT;
    else if(gt) return COMPARE_GT;
    return COMPARE_ER;
}

bool PatternMatch::is_multiple_usage(Computation* computation) {
    // note: that could be fused even it it multiple usage, just remain the one and fuse others, leave it a todo
    //false means cannot change computation, others may use computation
    return computation->out->related_idx.size()>1;
}

bool PatternMatch::is_same_affines(const vector<Affine> &a, const vector<Affine> &b, map<string,string>& m) {
    if(a.size()!=b.size()) return false;
    for(int i=0;i<a.size();i++){
        if(!Affine::is_same_pattern(a[i],b[i],m)) return false;
    }
    return true;
}

void PatternMatch::release(const vector<Computation *>& v) {
    for(auto& ptr:v){
        for(int i=0;i<ptr->num_in;i++){
             if(ptr->in[i]->sub_op_type!=DATA&&ptr->in[i]->tensorType!=WEIGHT)
                 ptr->in[i]->related_idx.erase(ptr->out->index);
        }        
    }
}

void PatternMatch::delete_t(const vector<Computation*>& v) {
    for(auto& ptr:v){
        delete ptr;
    }
}

bool PatternMatch::any2nop(int idx)  {
    /*
     * this pass could be time consuming, it's better triggered by other pass who can create nop
     */
    if(graph->computations[idx]->op!=DATA){
        auto top = graph->computations[idx];
        for(int i=0;i<top->num_in;i++){
            int index = top->in[i]->index;
            if(graph->computations[index]->op==NOP){
                auto nop = graph->computations[index];
                release({nop});
                //rebuild connection
                nop->in[0]->related_idx.insert(make_pair(idx,i));
                top->in[i] = nop->in[0];
                delete_t({nop});
                subst_history.emplace_back(__FUNCTION__ ,idx);
                return true;
            }
        }
    }
    return false;
}

bool PatternMatch::is_same_formula(Formula* left,Formula* right,map<string,string>&m) {
    if(left&&!right) return false;
    if(!left&&right) return false;
    if(!left&&!right) return true;
    if(left->autogen&&right->autogen) return true;
    if(!left->autogen&&!right->autogen) return Formula::is_same_pattern(*left,*right,m);
    return false;
}

bool PatternMatch::reverse_sum_mul_assign_both(int idx) {
    /*
     * merge two conv with same input into one conv
     * merge two matmul with same input into one conv
     * x sub_op_type is not a DATA,if we want to do that,do Nop to x and merge and remove Nop
     * action : add cond and split
     * double check
     */
    auto x = graph->computations[idx]->out;
    if(x->related_idx.size()==2){ // todo size>=2
        auto iter = x->related_idx.begin();
        auto assign_left = graph->computations[iter->first];
        auto assign_right = graph->computations[(++iter)->first];
        // comparing specific  is complicated,we just compare out_shape
        //temporary judgement
        if(!assign_left||!assign_right) return false;
        Compare assign_compare = compare_2_axis(assign_left->out->shape,assign_right->out->shape);
        if (assign_left->op==ASSIGN&&assign_right->op==ASSIGN&&assign_compare!=COMPARE_ER&&
                !is_multiple_usage(assign_left)&&!is_multiple_usage(assign_right)){
            auto mul_left = graph->computations[assign_left->out->related_idx.begin()->first];
            auto mul_right = graph->computations[assign_right->out->related_idx.begin()->first];
            Compare mul_compare = compare_2_axis(mul_left->out->shape,mul_right->out->shape);
            if(mul_left->op==MUL&&mul_right->op==MUL&&mul_compare!=COMPARE_ER&&
                    !is_multiple_usage(mul_left)&&!is_multiple_usage(mul_right)){
                auto sum_left = graph->computations[mul_left->out->related_idx.begin()->first];
                auto sum_right = graph->computations[mul_right->out->related_idx.begin()->first];
                Compare sum_compare = compare_2_axis(sum_left->out->shape,sum_right->out->shape);
                if(sum_left->op==SUM&&sum_right->op==SUM&&sum_compare!=COMPARE_ER&&
                   !is_multiple_usage(sum_left)&&!is_multiple_usage(sum_right)){
                    auto left = dynamic_cast<Sum*>(sum_left);
                    auto right = dynamic_cast<Sum*>(sum_right);
                    // add a concat and split
                    auto t = new Tensor, t1 = new Tensor,t2 = new Tensor;
                    //rebuild connection
                    for(auto &p:left->out->related_idx){
                        graph->computations[p.first]->in[p.second] = t1; //memory leak
                        t1->related_idx[p.first] = p.second;
                    }
                    left->out->related_idx.clear();
                    for(auto &p:right->out->related_idx){
                        graph->computations[p.first]->in[p.second] = t2; //memory leak
                        t2->related_idx[p.first] = p.second;
                    }
                    right->out->related_idx.clear();
                    //=====build finished=====
                    auto v = Graph::get_variable(graph->channel, left->subst.first.get_upper_bound() + right->subst.first.get_upper_bound(), 0, 1);
                    graph->push(new Cond(t,left->out,right->out,left->subst.first,right->subst.first,v,graph->num_cpt));
                    map<Variable,Affine> left_split = {make_pair(v,Affine(left->subst.first))};
                    map<Variable,Affine> right_split = {make_pair(v,Parser::make_affine({right->subst.first.name,"+",left->subst.first.get_upper_bound()}))};
                    graph->push(new Assign(t1,t,left->out->shape,graph->num_cpt,left_split));
                    graph->push(new Assign(t2,t,right->out->shape,graph->num_cpt,right_split));

                    subst_history.emplace_back(__FUNCTION__ ,idx);
                    return true;
                }
            }
        }
    }
    return false;
}

void PatternMatch::partition(Computation *computation,Computation* parent ,int &idx, bool group_new) const {
    static bool last_group_new=false;

    //if last_group_new is true, disable is_boundary
    if(group_new||is_boundary(computation,last_group_new)){
        computation->out->group_id.insert(idx++);

    }else{
        //otherwise ,idx is not increasing
        int index= computation->out->related_idx.begin()->first;
        auto &common=graph->computations[index]->out->group_id;
        computation->out->group_id.insert(common.begin(),common.end());
    }
    last_group_new = group_new;
    if(computation->out->group_id==parent->out->group_id) {
        computation->is_boundary = false;
    }

}

void PatternMatch::release_isolated_chain(TensorHandle t) {
    int idx = t->index;
    auto computation = graph->computations[idx];
    for(int i=0;i<computation->num_in;i++){
        int index = computation->in[i]->index;
        if(index!=-1){
            auto in = graph->computations[index];
            if(in->out->tensorType!=WEIGHT&&in->out->sub_op_type!=DATA&&!is_multiple_usage(in)) release_isolated_chain(in->out);
            if(in->out->tensorType!=WEIGHT&&in->out->sub_op_type!=DATA) {
                in->out->related_idx.erase(computation->out->index);
                //if(in->out->related_idx.empty()) {delete graph->computations[index];}
            }
        }

    }
    delete computation;
    graph->computations[idx] = nullptr;
}

int PatternMatch::get_contain_index(const vector<Variable>& shape, const Variable& v) {
    for(uint i=0;i<shape.size();i++){
        if(shape[i]==v) return i;
    }
    return -1;
}

bool PatternMatch::is_dim_expand(const vector<Variable> &initial, const vector<Variable> &next) {
    /*
     * todo we can tolerate dim_expand 0 1 2,for now it is 0
     */
    for(auto& x:next){
        if(get_contain_index(initial, x) == -1) return true;
    }
    return false;
}

bool PatternMatch::mul2assign_left(int idx) {
    if(graph->computations[idx]->op==MUL&& graph->computations[idx]->in[0]->sub_op_type == ASSIGN){
        int left_idx = graph->computations[idx]->in[0]->index;
        auto top = dynamic_cast<Mul*>(graph->computations[idx]);
        auto left = dynamic_cast<Assign*>(graph->computations[left_idx]);
        if(is_multiple_usage(left)) return false;
        if(top->in[1]->tensorType==WEIGHT&&left->in[0]->tensorType==MIX){
            for(auto&v:top->in[1]->shape){
                int index = get_contain_index(left->in[0]->shape,v);
                if(index==-1) return false;
            }
            //// assign in and out both contain means assign do not change those dim
            release({top,left});
            auto second = new Mul(left->out,left->in[0],top->in[1],left_idx,top->simplify_const_tensor_indices);
            auto first  = new Assign(top->out,left->out,top->out->shape,idx,left->lambda,left->formula);
            delete_t({top,left});
            graph->computations[idx] = first;
            graph->computations[left_idx] = second;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }
    return false;
}

bool PatternMatch::is_contain(const vector<Variable>& big,const vector<Variable>& small) {
    for(auto& p:small){
        if(get_contain_index(big, p) == -1) return false;
    }
    return true;
}

bool PatternMatch::is_overlap(const vector<Variable> &big, const vector<Variable> &small) {
    for(auto& p:small){
        if(get_contain_index(big, p) != -1) return true;
    }
    return false;
}

bool PatternMatch::add2assign_both(int idx) {
    if(graph->computations[idx]->op==ADD && graph->computations[idx]->in[0]->sub_op_type == ASSIGN &&
        graph->computations[idx]->in[1]->sub_op_type == ASSIGN){
        int left_idx = graph->computations[idx]->in[0]->index;
        int right_idx =graph->computations[idx]->in[1]->index;
        auto top = dynamic_cast<Add*>(graph->computations[idx]);
        auto down_left = dynamic_cast<Assign*>(graph->computations[left_idx]);
        auto down_right = dynamic_cast<Assign*>(graph->computations[right_idx]);
        //if add is not a elementwise-add ===> return false
        if(is_multiple_usage(down_left) || is_multiple_usage(down_right)||top->in[0]->shape!=top->in[1]->shape) return false;
        map<string,string> m;// empty mapping
        if(is_same_affines(down_left->lambda, down_right->lambda, m)&&
           is_same_formula(down_left->formula,down_right->formula,m)){
            release({top,down_left,down_right});
            auto new_down = new Add(down_left->out,down_left->in[0],down_right->in[0],left_idx);
            auto new_top = new Assign(top->out,down_left->out,top->out->shape,idx,down_left->lambda,down_left->formula);
            delete_t({top,down_left,down_right});
            graph->computations[idx] = new_top;
            graph->computations[left_idx] = new_down;
            graph->computations[right_idx] = nullptr;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }
    return false;
}

bool PatternMatch::mul2assign_both(int idx) {
    if(graph->computations[idx]->op==MUL && graph->computations[idx]->in[0]->sub_op_type == ASSIGN &&
       graph->computations[idx]->in[1]->sub_op_type == ASSIGN){
        int left_idx = graph->computations[idx]->in[0]->index;
        int right_idx =graph->computations[idx]->in[1]->index;
        auto top = dynamic_cast<Mul*>(graph->computations[idx]);
        auto down_left = dynamic_cast<Assign*>(graph->computations[left_idx]);
        auto down_right = dynamic_cast<Assign*>(graph->computations[right_idx]);
        //if add is not a elementwise-mul ===> return false
        if(is_multiple_usage(down_left) || is_multiple_usage(down_right)||top->in[0]->shape!=top->in[1]->shape) return false;
        map<string,string> m;// empty mapping
        if(is_same_affines(down_left->lambda, down_right->lambda, m)&&
           is_same_formula(down_left->formula,down_right->formula,m)){
            release({top,down_left,down_right});
            auto new_down = new Mul(down_left->out,down_left->in[0],down_right->in[0],left_idx);
            auto new_top = new Assign(top->out,down_left->out,top->out->shape,idx,down_left->lambda,down_left->formula);
            delete_t({top,down_left,down_right});
            graph->computations[idx] = new_top;
            graph->computations[left_idx] = new_down;
            graph->computations[right_idx] = nullptr;
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }
    return false;
}

bool PatternMatch::is_boundary(Computation *computation,bool last) {
    if(last) return false;
//    if(computation->op==MAX) return true;
    if(computation->op==SUM){
        auto item = dynamic_cast<Sum*>(computation);
        if(item->is_boundary) return true;
    }
    return false;
}

void PatternMatch::init_num_consumer(Computation* computation) const {
    //Assuming group_id is single
    auto group_id = computation->out->group_id;
    bool flag=true;
    for(auto&p:computation->out->related_idx){
        auto consumer = graph->computations[p.first];
        if(consumer->out->group_id==group_id) {flag=false;break;}
    }
    if(flag){
        computation->num_consumer = 0;
        computation->is_boundary = true;
    }else{
        computation->num_consumer = computation->out->related_idx.size();
    }
}

bool PatternMatch::reverse_both(int idx) {
    /*
     * merge two conv with same input into one conv
     * merge two matmul with same input into one conv
     * merge two other operator with same input into one
     * x sub_op_type is not a DATA,if we want to do that,do Nop to x and merge and remove Nop
     * action : add cond and split
     * double check
     */
    auto x = graph->computations[idx]->out;
//    if(idx==24) return false; //for debug
    if(x->related_idx.size()>=2){ // Assuming what we want to merge is at front in related_idx,if not ,leave it a todo
        auto iter = x->related_idx.begin();
        int left_index = iter->first;
        int right_index = (++iter)->first;
        Computation* left,*right;
        Computation* record_left= nullptr,*record_right= nullptr;
        while(true){
            left =graph->computations[left_index] ;
            right=graph->computations[right_index];
            if(!left||!right||left->op!=right->op||left==right) break;
            Compare compare_result = compare_2_axis(left->out->shape,right->out->shape);
            if(compare_result==COMPARE_ER) break;
            if(left->op==ASSIGN){
                auto assign_left = dynamic_cast<Assign*>(left);
                auto assign_right = dynamic_cast<Assign*>(right);
                map<string,string> m;// empty mapping
                if(!is_same_affines(assign_left->lambda, assign_right->lambda, m)|| !is_same_formula(assign_left->formula,assign_right->formula,m)) break;
            }
            if(left->op==RESHAPE) break;
            // record left and right
            record_left = left,record_right = right;
            if(is_multiple_usage(left)||is_multiple_usage(right)) break; // record and then break
            // update loop
            left_index = left->out->related_idx.begin()->first;
            right_index = right->out->related_idx.begin()->first;
        }
        if(record_left && record_right){
            auto t = new Tensor, t1 = new Tensor,t2 = new Tensor;
            //rebuild connection
            for(auto &p:record_left->out->related_idx){
                graph->computations[p.first]->in[p.second] = t1; //memory leak
                t1->related_idx[p.first] = p.second;
            }
            record_left->out->related_idx.clear();
            for(auto &p:record_right->out->related_idx){
                graph->computations[p.first]->in[p.second] = t2; //memory leak
                t2->related_idx[p.first] = p.second;
            }
            record_right->out->related_idx.clear();
            //=====build finished=====
            Variable a,b;
            if(record_left->op==SUM){
                a = dynamic_cast<Sum*>(record_left)->subst.first;
                b = dynamic_cast<Sum*>(record_right)->subst.first;
            }else{
                if(record_left->out->shape.size()<2 || record_right->out->shape.size() <2) return false;//TODO
                a = record_left->out->shape[1];
                b = record_right->out->shape[1];
            }
            auto v = Graph::get_variable(Graph::channel, a.get_upper_bound() + b.get_upper_bound(), 0, 1);
            graph->push(new Cond(t,record_left->out,record_right->out,a,b,v,graph->num_cpt));
            map<Variable,Affine> left_split = {make_pair(v,Affine(a))};
            map<Variable,Affine> right_split = {make_pair(v,Parser::make_affine({b.name,"+",a.get_upper_bound()}))};
            graph->push(new Assign(t1,t,record_left->out->shape,graph->num_cpt,left_split, nullptr));
            graph->push(new Assign(t2,t,record_right->out->shape,graph->num_cpt,right_split, nullptr));
            subst_history.emplace_back(__FUNCTION__ ,idx);
            return true;
        }
    }
    return false;
}

bool PatternMatch::reshape2assign_single(int idx) {
    //to remove reshape pseudo to create chance for assign2assign_single,
    //for other situation, reshape doesn't need to be removed,but we still did it
    //see reshape as assign, same as assign2assign_single
    if(graph->computations[idx]->op==RESHAPE&& graph->computations[idx]->in[0]->sub_op_type == ASSIGN) {
        int down_idx = graph->computations[idx]->in[0]->index;
        auto top = dynamic_cast<Reshape*>(graph->computations[idx]);
        auto down = dynamic_cast<Assign*>(graph->computations[down_idx]);
        int t_index=0,d_index=0;
        //assuming 'reshape' new shape!=original shape
        while(t_index<top->out->shape.size()&&d_index<down->out->shape.size()){
            if(top->out->shape[t_index]!=down->out->shape[d_index]) break;
            t_index++,d_index++;
        }
        int last_element=1;
        vector<int> multiplier;
        for(int i=top->out->shape.size()-1;i>=t_index;i--){
            multiplier.insert(multiplier.begin(),last_element);
            last_element*= top->out->shape[i].get_upper_bound();
        }
        vector<atomi> a;
        for(int i=t_index;i<top->out->shape.size();i++){
            a.emplace_back(top->out->shape[i].name);
            a.emplace_back("*");
            a.emplace_back(multiplier[i-t_index]);
            if(i!=top->out->shape.size()-1) a.emplace_back("+");
        }
        Affine affine = Parser::make_affine(a);
        vector<int> denominator;
        last_element = 1;
        for(int i=down->out->shape.size()-1;i>=d_index;i--){
            denominator.insert(denominator.begin(),last_element);
            last_element*= down->out->shape[i].get_upper_bound();
        }
        map<Variable,Affine> mapping;
        for(int i=d_index;i<down->out->shape.size()-1;i++){
            mapping[down->out->shape[i]] = affine.div(Affine(denominator[i-d_index]));
        }
        mapping[down->out->shape.back()] = affine.mod(Affine(down->out->shape.back().get_upper_bound()));

        // same as assign2assign_single
        for(auto&item:down->lambda){ //Affine
            for(auto&m:mapping){
                item.replace(m.first.name,m.second);
            }
        }
        if(down->formula){
            for(auto&m:mapping){
                down->formula->replace(m.first.name,m.second);
            }
        }

        release({top,down});
        auto new_top = new Assign(top->out,down->in[0],top->out->shape,idx,down->lambda,down->formula);
        delete_t({top,down});
        graph->computations[idx] = new_top;
        graph->computations[down_idx] = nullptr;
//        cout<<"##reshape2assign_result "<<new_top->to_compute()<<endl;
        subst_history.emplace_back(__FUNCTION__ ,idx);
        return true;
        //rebuild connection
//        for(auto &p:top->out->related_idx){
//            graph->computations[p.first]->in[p.second] = down->out; //memory leak
//            down->out->related_idx[p.first] = p.second;
//        }
//        down->out->related_idx.erase(top->out->index);
//        down->out->reshape(top->new_shape);
    }
    return false;
}
