//
// Created by sun on 2021/12/16.
//

#include "Affine.h"
#include"Producer.h"
#include<cassert>
#include<iostream>
#include <utility>
#include<algorithm>



bool Variable::check_name(const string &name) {
    assert(!name.empty());

    if (!isalpha(name[0]) && name[0] != '_') return false;
    for (int i = 1; i < name.size(); i++) {
        if (!isalnum(name[i]) && name[i] != '_') return false;
    }
    return true;
}




void Affine::update(const string& key, int val) {
    //if(val==0&&coefficients.find(key)!=coefficients.end()) {coefficients.erase(key);return ;}
    coefficients[key] += val;
    if(coefficients[key]==0) coefficients.erase(key);
}
void Affine::update(const pair<int,Affine>& key, int val){
    //if(val==0&& divider.find(key)!=divider.end()) {divider.erase(key);return ;}
    divider[key]+=val;
    if(divider[key]==0) divider.erase(key);
}

[[maybe_unused]] void Affine::erase(const string& key){
    coefficients.erase(key);
}
void Affine::erase(const pair<int, Affine> &key) {
    divider.erase(key);
}
Affine Affine::add(const Affine& that) const {
    Affine ans;
    for(auto& m:this->coefficients){
        ans.update(m.first, m.second);
    }
    for(auto& m:that.coefficients){
        ans.update(m.first, m.second);
    }
    for(auto&m:this->divider){
        ans.update(m.first, m.second);
    }
    for(auto&m:that.divider){
        ans.update(m.first, m.second);
    }
    ans.variables.insert(this->variables.begin(),this->variables.end());
    ans.variables.insert(that.variables.begin(),that.variables.end());
    return ans;
}

void Affine::merge(const Affine& that,int multiple){
    // same as add function
    for(auto&m:that.coefficients){
        this->update(m.first, m.second*multiple);
    }
    for(auto&m:that.divider){
        this->update(m.first, m.second*multiple);
    }
    this->variables.insert(that.variables.begin(),that.variables.end());
}
Affine Affine::sub(const Affine &that)const {
    Affine ans;
    for(auto& m:this->coefficients){
        ans.update(m.first, m.second);
    }
    for(auto& m:that.coefficients){
        ans.update(m.first, -m.second);
    }
    for(auto&m:this->divider){
        ans.update(m.first, m.second);
    }
    for(auto&m:that.divider){
        ans.update(m.first, -m.second);
    }
    ans.variables.insert(this->variables.begin(),this->variables.end());
    ans.variables.insert(that.variables.begin(),that.variables.end());
    return ans;
}

Affine Affine::mul(const Affine&that)const {
    //special case : mul by zero
    assert(this->type==AFFINE_CONSTANT||that.type==AFFINE_CONSTANT);
    int value;
    Affine ans;
    if(this->type==AFFINE_CONSTANT){
        value = this->coefficients.at("");
        if(value==0) ans.update("", 0);
        else{
            for(auto& m:that.coefficients){
                ans.update(m.first, m.second * value);
            }
            for(auto& m:that.divider){
                ans.update(m.first, m.second * value);
            }
        }
    }else if(that.type==AFFINE_CONSTANT){
        value = that.coefficients.at("");
        if(value==0) ans.update("", 0);
        else{
            for(auto& m:this->coefficients){
                ans.update(m.first, m.second * value);
            }
            for(auto& m:this->divider) {
                ans.update(m.first, m.second * value);
            }
        }
    }
    ans.variables.insert(this->variables.begin(),this->variables.end());
    return ans;
}

Affine::Affine(uint value):type(AFFINE_CONSTANT) {
    coefficients[""] = value;
    //coefficients.insert(coefficients.begin(),make_pair("",value));
}

Affine::Affine(const Variable &v):type(AFFINE_VARIABLE) {
    coefficients[v.name] = 1;
    variables.insert(v.name);
}

Affine Affine::neg() const{
    //this function is rarely used,case we have sub
    //if we delete sub,we can use neg
    Affine ans;
    for(auto& m:this->coefficients){
        ans.update(m.first, -m.second);
    }
    for(auto&m:this->divider){
        ans.update(m.first, -m.second);
    }
    ans.variables.insert(this->variables.begin(),this->variables.end());
    return ans;
}

bool Affine::is_zero() {
    if(type!=AFFINE_CONSTANT)  return false;
    return coefficients[""]==0;
}

Affine Affine::div(const Affine& that) {
    //special case 1  (h/3 + (-h)/3)/5 ===> h/3 + (-h)/3
    //special case 2*h/4 ==> h/2 --check
    assert(that.type==AFFINE_CONSTANT);
    assert(that.coefficients.at("")>0); //factor needs to be positive

    int factor = that.coefficients.at("");
    if(factor==1)   return *this;//copy this

    Affine ans = this->normalize(factor); // ans hold quotient and this keep remainder
    int denominator = get_denominator();
    if(is_special_case()) ans.merge(*this);// copy this
    else if(denominator==INT32_MAX) // this not containing divider
        ans.update(make_pair(factor, *this), 1);
    else{
        // denominator * factor
        for(auto&m:this->coefficients){
            m.second *=denominator;
        }
        for(auto iter=divider.begin();iter!=divider.end();iter++){
            if(iter==iterator){
                this->merge(iter->first.second,iter->second);
                this->erase(iter->first);
            }else{
                iter->second *= denominator;
            }
        }
        factor*= denominator;
        ans.merge(normalize(factor));
        ans.update(make_pair(factor, *this), 1);// copy this, not sharing memory with this
    }
    ans.variables.insert(this->variables.begin(),this->variables.end());
    return ans;
}
Affine Affine::mod(const Affine& that){
    assert(that.type==AFFINE_CONSTANT);
    int factor = that.coefficients.at("");
    assert(factor!=0); //divide by 0
    if(factor==1) return Affine(0);
    //// a mod b === a - a/b *b
    auto tmp = this->div(that).mul(that);
    auto ans = this->sub(tmp);
    return ans;
}

string Affine::to_string(bool te) const {
    /*
     * if type == AFFINE_VARIABLE return variable name
     * string contain % which cannot be represented by coefficient
     */
    string ans;
    vector<string> tmp;

    Affine duplication = *this;
    for(const auto& m: duplication.divider){
        if(m.first.first==m.second){
            duplication.merge(m.first.second);
            if(te){
                tmp.push_back("te.indexmod("+m.first.second.to_string(te)+","+::to_string(m.first.first)+")");
            }else{
                tmp.push_back("("+m.first.second.to_string(te)+")%"+::to_string(m.first.first));
            }
            duplication.erase(m.first);//erase
        }else if(m.first.first==-m.second){
            duplication.merge(m.first.second,-1);
            if(te){
                tmp.push_back("te.indexmod("+m.first.second.to_string(te)+","+::to_string(m.first.first)+")");
            }else{
                tmp.push_back("("+m.first.second.to_string(te)+")%"+::to_string(m.first.first));
            }
            duplication.erase(m.first);//erase
        }else{
            string instance;
            if(te){
                instance = "te.indexdiv("+m.first.second.to_string(te)+","+::to_string(m.first.first)+")";
                if(m.second!=1) instance+= "*"+::to_string(m.second);
            }else{
                instance+="("+m.first.second.to_string()+")"+"/"+::to_string(m.first.first);
                if(m.second!=1) instance+= "*"+::to_string(m.second);
            }
            tmp.push_back(instance);
        }
    }
    for(const auto& m:duplication.coefficients){
        string instance;
        if(m.first.empty())instance+= ::to_string(m.second);
        else if(m.second!=1) instance+= ::to_string(m.second)+"*"+m.first;
        else instance+=m.first;
        tmp.push_back(instance);
    }
    for(int i=0;i<tmp.size();i++){
        ans += tmp[i];
        if(i!=tmp.size()-1) ans+="+";
    }
    if(tmp.empty()) ans+="0";
    return ans;


//    int i=0,c_size = coefficients.size(),d_size = divider.size();
//    for(const auto& m:coefficients){
//        //Assuming m.second!=0
//        if(m.first.empty())ans+= ::to_string(m.second);
//        else if(m.second!=1) ans+= ::to_string(m.second)+"*"+m.first;
//        else ans+=m.first;
//        if(i!=c_size-1) ans+="+";
//        i++;
//    }
//
//    if(!divider.empty()){
//        if(i!=0) {ans+="+";i=0;}
//        for(const auto& m: divider){
//            ans+="("+m.first.second.to_string()+")";
//            if(m.first.first!=1) ans+="/"+::to_string(m.first.first);
//            if(m.second!=1) ans+= "*"+::to_string(m.second);
//            if(i!=d_size-1) ans+="+";
//            i++;
//        }
//    }
//    if(coefficients.empty()&&divider.empty()) ans+="0"; // special case
//    return ans;
}
bool Affine::check(const map<string, int> &eval_vap) const {
//    for(auto&s:variables){
//        if(eval_vap.find(s)==eval_vap.end()) return false;
//    }
//    return true;
    return std::all_of(variables.begin(),variables.end(),[&eval_vap](auto& item){return eval_vap.find(item)!=eval_vap.end();});
}

int Affine::evaluate(const map<string, int> &eval_vap) const {
    assert(check(eval_vap));
    int ans=0;
    for(auto&m:coefficients){
        if(!m.first.empty()) ans += eval_vap.at(m.first)*m.second;
        else ans+= m.second;
    }
    for(auto& m:divider){
        int value = m.first.second.evaluate(eval_vap);
        ans+= Math::floordiv(value,m.first.first)*m.second;
    }
    return ans;
}
Affine Affine::normalize(int& factor){
    /*
     *  separate this/factor into quotient and remainder(this)
     *  update factor with factor / gcd
     */
    Affine quotient;
    vector<int> v;
    for(auto &m:this->coefficients){
        v.push_back(m.second);
        int value = Math::floordiv(m.second+(factor-1)/2,factor);
        if(value!=0){
            quotient.update(m.first, value);
            this->update(m.first, -factor * value);
        }
    }
    for(auto& m:this->divider){
        v.push_back(m.second);
        int value = Math::floordiv(m.second+(factor-1)/2,factor);
        if(value!=0) {
            quotient.update(m.first, value);
            this->update(m.first, -factor * value);
        }
    }
    int gcd = Math::greatest_common_divisor(v,factor);
    if(gcd==1||gcd<0) return quotient;
    for(auto &m:this->coefficients){
        m.second/=gcd;
    }
    for(auto&m:this->divider){
        m.second/=gcd;
    }
    factor/=gcd;
    return quotient;
}
int Affine::get_denominator() {
    /*
     *  get minimum denominator
     */
    int ans = INT32_MAX;
    for (auto iter=divider.begin();iter!=divider.end();iter++) {
        if (iter->first.first < ans) {
            ans = iter->first.first;
            iterator = iter;
        }
    }
    return ans;
}

bool Affine::is_special_case() {
    if(!coefficients.empty()) return false;
    if(divider.size()!=2) return false;
    auto iter = divider.begin();
    auto &item1 = *iter++;
    auto &item2 = *iter;
    if(item1.first.first!=item2.first.first) return false;
    if(item1.second!=item2.second) return false;
    if( is_opposite(item1.first.second,item2.first.second)){
        if(item1.second>0) item1.second = 1,item2.second=1;
        else this->divider.clear();
        return true;
    }
    return false;
}

bool Affine::is_opposite(Affine a, Affine b){
    //double pointer
    //compare coefficient
    auto iter1 = a.coefficients.begin();
    auto iter2 = b.coefficients.begin();
    while(iter1!=a.coefficients.end()&&iter2!=b.coefficients.end()){
        if(iter1->first!=iter2->first) return false;
        if(iter1->second!=-iter2->second) return false;
        iter1++;
        iter2++;
    }
    if(iter1==a.coefficients.end()&&iter2!=b.coefficients.end()) return false;
    if(iter1!=a.coefficients.end()&&iter2==b.coefficients.end()) return false;
    // compare divider
    auto iter3 = a.divider.begin();
    auto iter4 = b.divider.begin();
    while(iter3!=a.divider.end()&&iter4!=b.divider.end()){
        if(iter3->first!=iter4->first) return false;
        if(iter3->second!=-iter4->second) return false;
        iter3++;
        iter4++;
    }
    if(iter3!=a.divider.end()&&iter4==b.divider.end()) return false;
    if(iter3==a.divider.end()&&iter4!=b.divider.end()) return false;
    return true;
}

bool Affine::operator==(const Affine &that) const {
    auto iter1 = this->coefficients.begin();
    auto iter2 = that.coefficients.begin();
    while(iter1!=this->coefficients.end()&&iter2!=that.coefficients.end()){
        if(iter1->first!=iter2->first) return false;
        if(iter1->second!=iter2->second) return false;
        iter1++;
        iter2++;
    }
    if(iter1==this->coefficients.end()&&iter2!=that.coefficients.end()) return false;
    if(iter1!=this->coefficients.end()&&iter2==that.coefficients.end()) return false;
    // compare divider
    auto iter3 = this->divider.begin();
    auto iter4 = that.divider.begin();
    while(iter3!=this->divider.end()&&iter4!=that.divider.end()){
        if(iter3->first!=iter4->first) return false;
        if(iter3->second!=iter4->second) return false;
        iter3++;
        iter4++;
    }
    if(iter3!=this->divider.end()&&iter4==that.divider.end()) return false;
    if(iter3==this->divider.end()&&iter4!=that.divider.end()) return false;
    return true;
}

bool Affine::operator<(const Affine &that) const {
    if(coefficients.size()!=that.coefficients.size()) return coefficients.size()<that.coefficients.size();
    auto iter1 = this->coefficients.begin();
    auto iter2 = that.coefficients.begin();
    while(iter1!=this->coefficients.end()&&iter2!=that.coefficients.end()){
        if(iter1->first!=iter2->first) return iter1->first < iter2->first;
        if(iter1->second!=iter2->second) return iter1->second < iter2->second;
        iter1++;
        iter2++;
    }
    if(divider.size()!=that.divider.size()) return divider.size()<that.divider.size();
    auto iter3 = this->divider.begin();
    auto iter4 = that.divider.begin();
    while(iter3!=this->divider.end()&&iter4!=that.divider.end()){
        if(iter3->first!=iter4->first) return iter3->first < iter4->first;
        if(iter3->second!=iter4->second) return iter3->second < iter4->second;
        iter3++;
        iter4++;
    }
    // if *this==that false
    return false;
}

void Affine::replace(const string& source,const Affine& target) {
    //Assuming target is normalized
    if(variables.find(source)==variables.end()) return;
    if(coefficients.find(source)!=coefficients.end()){//found
        int value = coefficients[source];
        coefficients.erase(source);
        merge(target,value);
        this->variables.erase(source);
        this->variables.insert(target.variables.begin(),target.variables.end());
    }
    for(auto &m:divider){
        auto& tmp = const_cast<Affine&>(m.first.second);
        tmp.replace(source,target);
        int factor = m.first.first;
        this->merge(tmp.normalize(factor));
    }
}

bool Affine::is_same_pattern(const Affine &left, const Affine &right, map<string, string>& m) {
    //if nothing be contained in m, this function is equal to operator==
    auto iter1 = left.coefficients.begin();
    auto iter2 = right.coefficients.begin();
    while(iter1!=left.coefficients.end()&&iter2!=right.coefficients.end()){
        if(!is_same_string(iter1->first,iter2->first,m)) return false;
        if(iter1->second!=iter2->second) return false;
        iter1++;
        iter2++;
    }
    if(iter1==left.coefficients.end()&&iter2!=right.coefficients.end()) return false;
    if(iter1!=left.coefficients.end()&&iter2==right.coefficients.end()) return false;
    // compare divider
    auto iter3 = left.divider.begin();
    auto iter4 = right.divider.begin();
    while(iter3!=left.divider.end()&&iter4!=right.divider.end()){
        if(iter3->first.first!=iter4->first.first||!is_same_pattern(iter3->first.second,iter4->first.second,m)) return false;
        if(iter3->second!=iter4->second) return false;
        iter3++;
        iter4++;
    }
    if(iter3!=left.divider.end()&&iter4==right.divider.end()) return false;
    if(iter3==left.divider.end()&&iter4!=right.divider.end()) return false;
    return true;
}

bool Affine::is_same_string(const string &a, const string &b, map<string, string> &m) {
    if(a==b) return true;
    if(m.find(a)==m.end()) return false;
    return m[a]==b;
}

bool Affine::is_pure_divider(Affine &affine, uint& numerator) {
    if(coefficients.empty()&&divider.size()==1&&divider.begin()->second==1) {
        affine = divider.begin()->first.second;// do not change affine
        numerator = divider.begin()->first.first;
        return true;
    }
    return false;
}

bool Affine::is_coefficients_equals(int value) {
    if(!divider.empty()) return false;
    for(auto&m:coefficients){
        if(!m.first.empty()&&m.second!=value) return false;
    }
    return true;
}

bool Affine::operator!=(const Affine &that) const {
    return !(*this==that);
}

Affine::Affine(const Affine &that) {
    this->coefficients = that.coefficients;
    this->divider = that.divider;
    this->type = that.type;
    this->variables = that.variables;
}

Affine Affine::ceil() {
    //note :use const_cast, does it have side effects?
    Affine ans = *this;
    for(auto& d:ans.divider){
        const_cast<Affine&>(d.first.second).update("",d.first.first-1);
    }
    return ans;
}


Variable::Variable(const string &name) {
    assert(check_name(name));
    this->name = name;
}

int Variable::get_upper_bound() const {
    if(upper_bound.is_concrete) return upper_bound.value;
    return 0;
}

int Variable::get_lower_bound() const {
    if(lower_bound.is_concrete) return lower_bound.value;
    return 0;
}

Variable::Variable(const string& name, int _upper, int _lower):lower_bound(_lower),upper_bound(_upper) {
    check_name(name);
    this->name = name;
}

Variable::Variable(const string& name,string _upper, string _lower):lower_bound(std::move(_lower)),upper_bound(std::move(_upper)) {
    check_name(name);
    this->name = name;
}

Variable::Variable(const Variable &v) {
    this->name = v.name;
    this->upper_bound = v.upper_bound;
    this->lower_bound = v.lower_bound;
}

bool Variable::operator!=(const Variable &v) const {
    return this->name!=v.name;
}

bool Variable::operator==(const Variable &v) const {
    return this->name==v.name;
}

bool Variable::operator<(const Variable &v) const {
    //Assuming lower_bound both are 0
    return this->name < v.name;
}

string Variable::to_string() const {
    string result = "for(int "+name+"="+lower_bound.to_string()+";"
            +name+"<"+upper_bound.to_string()+";++"+name+")";
    return result;
}

[[maybe_unused]] bool Variable::lt_bound(const Variable &v) const {
    return this->get_upper_bound()<v.get_upper_bound();
}

string Bound::to_string() const {
    if(is_concrete) return ::to_string(value);
    return bound_name;
}
