//
// Created by sun on 2021/12/20.
//

#include "Formula.h"
#include<cassert>
#include<stack>
#include<iostream>
#include <utility>
Constraint::Constraint(AffineConstraint c, const Affine &l, const Affine &r):constraint(c),left(l),right(r) {
    uint numerator;
    Affine temp;
    // special case : i/3 <5  ===> i<15
    if(left.is_pure_divider(temp,numerator)&&right.type==AFFINE_CONSTANT){
        left = temp;
        right = r.mul(Affine(numerator));
    }else  if(left.type==AFFINE_CONSTANT&&right.is_pure_divider(temp,numerator)){
        left = l.mul(Affine(numerator));
        right = temp;
    }
}

bool Constraint::evaluate(const map<string, int> &eval_map) const {
    int lvalue = left.evaluate(eval_map);
    int rvalue = right.evaluate(eval_map);
    switch (constraint) {
        case CONSTRAINT_EQ:
            return lvalue==rvalue;
        case CONSTRAINT_GE:
            return lvalue>=rvalue;
        case CONSTRAINT_GT:
            return lvalue>rvalue;
        case CONSTRAINT_LE:
            return lvalue<=rvalue;
        case CONSTRAINT_LT:
            return lvalue<rvalue;
        case CONSTRAINT_NE:
            return lvalue!=rvalue;
        default:
            //assert(false);
            return false;
    }
}

string Constraint::to_string(bool te) const {
    //cout << left << ' ' << right << endl;
    return left.to_string(te)+affine_constraint_string[constraint]+right.to_string(te);
}

void Constraint::replace(const string &source, const Affine &target) {
    if(left.type!=AFFINE_CONSTANT) left.replace(source,target);
    if(right.type!=AFFINE_CONSTANT) right.replace(source,target);
}

bool Constraint::operator==(const Constraint& that) const {
    /*
     * exactly same, do not transform
     */
    if(this->left!=that.left) return false;
    if(this->constraint!=that.constraint) return false;
    if(this->right!=that.right) return false;
    return true;
}

bool Constraint::is_same_pattern(const Constraint& a,const Constraint& b,map<string,string>& m) {
    /*
     * exactly same, do not transform
     */
    if(!Affine::is_same_pattern(a.left,b.left,m)) return false;
    if(a.constraint!=b.constraint) return false;
    if(!Affine::is_same_pattern(a.right,b.right,m)) return false;
    return true;
}

bool Constraint::operator!=(const Constraint &that) const {
    return !(*this==that);
}

Formula::Formula(LogicSymbol symbol, Formula *left, Formula *right):symbol(symbol),left(left),right(right),entity() {
    if(left->same_symbol==ENTITY_CONSTRAINT&&right->same_symbol==ENTITY_CONSTRAINT) this->same_symbol = symbol;
    else if(left->same_symbol==ENTITY_CONSTRAINT&&right->same_symbol==symbol) this->same_symbol = symbol;
    else if(right->same_symbol==ENTITY_CONSTRAINT&&left->same_symbol==symbol) this->same_symbol = symbol;
    else if(left->same_symbol==right->same_symbol&&left->same_symbol==symbol) this->same_symbol = symbol;
}

Formula::Formula(Constraint c):symbol(ENTITY_CONSTRAINT),entity(std::move(c)) {

}

Formula *Formula::create_all(const vector<Constraint> &v) {
    if(v.empty()) return nullptr;
    auto ans = new Formula(v[0]);
    for(int i=1;i<v.size();i++){
        auto that = new Formula(v[i]);
        ans = new Formula(LOGIC_CONJUNCTION,ans,that);
    }
    return ans;
}

Formula *Formula::create_any(const vector<Constraint> &v) {
    if(v.empty()) return nullptr;
    auto ans = new Formula(v[0]);
    for(int i=1;i<v.size();i++){
        auto that = new Formula(v[i]);
        ans = new Formula(LOGIC_DISJUNCTION,ans,that);
    }
    return ans;
}

Formula *Formula::create_general(const vector<mix> &postfix) {
    //Assuming postfix is legal
    //Assuming symbol is not same,so same_symbol=LOGIC_INVALID
    if(postfix.empty()) return nullptr;
    stack<Formula*> s;
    for(auto& p:postfix){
        if(p.index()==0) s.push(new Formula(std::get<0>(p)));
        else{
            auto symbol = std::get<1>(p);
            switch (symbol) {
                case LOGIC_NEGATION:{
                        assert(!s.empty());
                        auto r = s.top();
                        s.pop();
                        s.push(new Formula(symbol, nullptr,r));
                        break;
                    }

                case LOGIC_CONJUNCTION:
                case LOGIC_DISJUNCTION:
                case LOGIC_IMPLICATION:
                case LOGIC_EQUIVALENCE:
                    do_binary(s,symbol);
                    break;
                case QUANTIFIER_ALL:
                case QUANTIFIER_EXIST:
                    cerr<< "not supported yet"<<endl;
                    break;
                default:
                    assert(false);
            }
        }
    }
    return s.top();
}

void Formula::do_binary(stack<Formula *> &s, LogicSymbol symbol1) {
    assert(s.size()>=2);
    auto r = s.top();
    s.pop();
    auto l = s.top();
    s.pop();
    auto ans = new Formula(symbol1,l,r);
    s.push(ans);
}

bool Formula::evaluate(const map<string,int> &eval_map) const {
    // this is root of expression tree
    bool l=true,r=true;
    if(left) l = left->evaluate(eval_map);
    if(right) r = right->evaluate(eval_map);
    switch (symbol) {
        case LOGIC_NEGATION:
            return !r;
        case LOGIC_CONJUNCTION:
            return l&&r;
        case LOGIC_DISJUNCTION:
            return l||r;
        case LOGIC_IMPLICATION:
            return !l||r;
        case LOGIC_EQUIVALENCE:
           return l==r;
        case ENTITY_CONSTRAINT:
            return entity.evaluate(eval_map);
        case QUANTIFIER_ALL:
        case QUANTIFIER_EXIST:
            cerr<< "not supported yet"<<endl;
            break;
        default:
            break;
    }
    return false;
}

Formula *Formula::conjunction(Formula* that) {
    if(!that) return this;
    if(this->autogen&&that->autogen) return  this;
    return new Formula(LOGIC_CONJUNCTION,this,that);
}

Formula *Formula::disjunction(Formula* that) {
    if(!that) return this;
    if(this->autogen&&that->autogen) return  this;
    return new Formula(LOGIC_DISJUNCTION,this,that);
}

string Formula::to_same_string(bool te) const {
    string result;
    if(left) result+= left->to_same_string(te);
    if(this->symbol==ENTITY_CONSTRAINT) result+=this->entity.to_string(te)+",";
    if(right) result+= right->to_same_string(te);
    return result;
}

string Formula::to_string(bool te) const {
    string str = to_same_string(te);
    str.erase(str.end() - 1);
    if(same_symbol==LOGIC_CONJUNCTION) return "te.all("+str+")";
    if(same_symbol==LOGIC_DISJUNCTION) return "te.any("+str+")";
    //return c-style formula string
    return to_string_c_style();
}

string Formula::to_string_c_style(bool te) const {
    string result;
    if(left) result+= "("+ left->to_string_c_style(te) + ")";
    if(this->symbol==ENTITY_CONSTRAINT) result+=this->entity.to_string();
    else result+=logic_symbol_string[symbol];
    if(right) result+= "("+ right->to_string_c_style(te) +")";
    return result;
}

void Formula::replace(const string &source, const Affine &target) {
    //medium travel
    if(left) left->replace(source,target);
    if(this->symbol==ENTITY_CONSTRAINT) entity.replace(source,target);
    if(right) right->replace(source,target);
}

bool Formula::operator==(const Formula& that) const{
    /*
     * exactly same, do not transform
     */
    if(left&&!that.left|| !left&&that.left) return false;
    if(right&&!that.right|| !right&&that.right) return false;
    if(this->symbol!=that.symbol) return false;
    if(this->symbol==ENTITY_CONSTRAINT&&this->entity!=that.entity) return false;

    if(left&&that.left&&*left!=*that.left) return false;
    if(right&&that.right&&*right!=*that.right) return false;
    return true;
}

bool Formula::operator!=(const Formula &that) const {
    return !(*this==that);
}

bool Formula::is_same_pattern(const Formula& a,const Formula& that,map<string,string>& m) {
    if(a.left&&!that.left|| !a.left&&that.left) return false;
    if(a.right&&!that.right|| !a.right&&that.right) return false;
    if(a.symbol!=that.symbol) return false;
    if(a.symbol==ENTITY_CONSTRAINT&&!Constraint::is_same_pattern(a.entity,that.entity,m)) return false;

    if(a.left&&that.left&&!is_same_pattern(*a.left,*that.left,m)) return false;
    if(a.right&&that.right&&!is_same_pattern(*a.right,*that.right,m)) return false;
    return true;
}

//FormulaOpBase::FormulaOpBase(LogicSymbol symbol1, Constraint *c)
//        :num_parameter(1),symbol(symbol1),left(nullptr),right(c) {
//
//}
//
//FormulaOpBase::FormulaOpBase(LogicSymbol symbol1, Constraint *c, Constraint *c1)
//        :num_parameter(2),symbol(symbol1),left(c),right(c1){
//}
//
//FormulaNeg::FormulaNeg(Constraint *c):FormulaOpBase(LOGIC_NEGATION,c) {
//
//}
//
//FormulaBinary::FormulaBinary(LogicSymbol symbol1, Constraint *c, Constraint *c1) : FormulaOpBase(symbol1, c, c1) {
//
//}
