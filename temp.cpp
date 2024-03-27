#include <bits/stdc++.h>
using namespace std;
#define rep(i, n) for (int i = 0; i < (int)(n); i++)
using ll = long long;

int main() {
  int N;
  cin >> N;
  
  vector<tuple<string, int, int>> tpl;
  rep(i,N){
    string s;
    int a;
    cin >> s >> a;
    tpl.push_back(make_tuple(s, a, i+1));
  }
  
  sort(tpl.begin(), tpl.end());
  
  vector<tuple<int, string, int>> tpl2;
  rep(i,N-1){
    string s;
    int a;
    int j;
    tie(s, a, j) = tpl[i];
    tpl2.push_back(make_tuple(a, s, j));
  }
  
  sort(tpl2.rbegin(), tpl2.rend());
  
  rep(i, N) cout <<  get<2>(tpl2[i]) << '\n';
  
  return 0;
}
