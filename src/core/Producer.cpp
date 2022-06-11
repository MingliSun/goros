//
// Created by sun on 2022/1/6.
//

#include "Producer.h"
#include<cassert>
int Producer::var_num = 0;
int Producer::tensor_num = 0;
int Producer::fused_num = 0;
vector<vector<int>> Producer::var_debug_num = std::move(vector<vector<int>>(2,vector<int>(5)));

string Producer::get_unique_variable_name(){
    return "v"+to_string(var_num++);
}

string Producer::get_unique_tensor_name() {
    return "tensor_"+to_string(tensor_num++);
}

string Producer::get_debug_variable_name(int row, int column) {
    string result = variable_name[row][column];
    if(var_debug_num[row][column]!=0)  result+=to_string(var_debug_num[row][column]);
    var_debug_num[row][column]++;
    return result;
}
string Producer::get_fused_operator_name() {
    return "fused_"+to_string(fused_num++);
}

vector<int> Math::get_concrete_index(const vector<Variable> &vars, int n) {
    int len = vars.size();
    vector<int> result(len);
    int i = len-1;
    while(n!=0){
        result[i] = n%(vars[i].get_upper_bound());
        n = n/vars[i].get_upper_bound();
        i--;
    }
    return result;
}

int Math::get_concrete_index(const vector<Variable> &vars, const vector<int> &integers) {
    int stride = 1;
    int result = 0;
    assert(vars.size()==integers.size());
    for(int i=(int)integers.size()-1;i>=0;i--){
        result+= stride*integers[i];
        stride *= vars[i].get_upper_bound();
    }
    return result;
}

vector<int> Math::get_concrete_index(const vector<Affine>& exprs, const vector<Variable>& vars, const vector<int>& integers) {
    int len = exprs.size();
    vector<int> result;
    map<string,int> m;
    assert(vars.size()==integers.size());
    for(int i=0;i<vars.size();i++){
        m[vars[i].name] = integers[i];
    }
    result.reserve(exprs.size());
    for(auto&expr:exprs){
        result.push_back(const_cast<Affine&>(expr).evaluate(m));
    }
    return result;
}

vector<int> Math::get_concrete_index(const vector<Variable> &out, const vector<Variable> &in, const vector<int>& src) {
    vector<int> result;
    map<Variable,int> m;
    assert(out.size()==src.size());
    for(int i=0;i<out.size();i++){
        m[out[i]] = src[i];
    }
    result.reserve(in.size());
    for(auto&p:in){
        result.push_back(m[p]);
    }
    return result;
}

int Math::greatest_common_divisor(int x,int y){
    int ans = y;
    while(x%y!=0){
        ans = x%y;
        x = y;
        y = ans;
    }
    return ans;
}
int Math::greatest_common_divisor(const vector<int>& v,int b){
    assert(!v.empty());
    int ans=INT32_MAX;
    for(auto& i:v){
        ans = min(ans,greatest_common_divisor(i,b));
        if(ans==1) return ans;
    }
    return ans;
}

int Math::modulo(int a, int b) {
    return (a % b + b) % b;
}

int Math::floordiv(int a, int b) {
    return (a - modulo(a, b)) / b;
}

float Math::modulo(float a, float b) {
    float quotient = floor(a/b);
    return a - b*quotient;
}
