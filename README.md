# Design and Analysis of Algorithms — Lab Manual

**Vivaan Singh Adhikari · 24/48029**

---

## Lab 1: Randomized Quick Sort

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static long comparisons = 0;

void swap(int* a, int* b) {
    int t = *a;
    *a = *b;
    *b = t;
}

int random_int(int p, int r) {
    return p + rand() % (r - p + 1);
}

int partition(int A[], int p, int r) {
    int x = A[r];
    int i = p - 1;

    for (int j = p; j < r; j++) {
        comparisons++;
        if (A[j] <= x) {
            i++;
            swap(&A[i], &A[j]);
        }
    }
    swap(&A[i + 1], &A[r]);
    return i + 1;
}

int randomized_partition(int A[], int p, int r) {
    int i = random_int(p, r);
    swap(&A[r], &A[i]);
    return partition(A, p, r);
}

void randomized_quicksort(int A[], int p, int r) {
    if (p < r) {
        int q = randomized_partition(A, p, r);
        randomized_quicksort(A, p, q - 1);
        randomized_quicksort(A, q + 1, r);
    }
}

int main(void) {
    srand((unsigned)time(NULL));

    int A[] = { 4, 2, 5, 3, 1, 9, 8, 7, 6, 0 };
    int n = (int)(sizeof(A) / sizeof(A[0]));

    randomized_quicksort(A, 0, n - 1);

    for (int i = 0; i < n; i++)
        printf("%d ", A[i]);
    putchar('\n');

    printf("comparisons: %ld\n", comparisons);

    return 0;
}
```

---

## Lab 2: Randomized Select

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int randomizedPartition(int A[], int p, int r) {
    int i = p + rand() % (r - p + 1);
    int temp = A[i];
    A[i] = A[r];
    A[r] = temp;

    int pivot = A[r];
    int j = p - 1;
    for (i = p; i < r; i++) {
        if (A[i] <= pivot) {
            j++;
            temp = A[j];
            A[j] = A[i];
            A[i] = temp;
        }
    }
    temp = A[j + 1];
    A[j + 1] = A[r];
    A[r] = temp;
    return j + 1;
}

int randomizedSelect(int A[], int p, int r, int i) {
    if (p == r)
        return A[p];

    int q = randomizedPartition(A, p, r);
    int k = q - p + 1;

    if (i == k)
        return A[q];
    else if (i < k)
        return randomizedSelect(A, p, q - 1, i);
    else
        return randomizedSelect(A, q + 1, r, i - k);
}

int main() {
    srand(time(NULL));

    int A[] = {3, 2, 9, 0, 7, 5, 4, 8, 6, 1};
    int n = sizeof(A) / sizeof(A[0]);

    for (int i = 1; i <= n; i++) {
        int result = randomizedSelect(A, 0, n - 1, i);
        printf("The %d-th smallest element is %d\n", i, result);
    }

    return 0;
}
```

---

## Lab 3: Kruskal's Algorithm

```c
// C code to implement Kruskal's algorithm
#include <stdio.h>
#include <stdlib.h>

// Comparator function to use in sorting
int comparator(const void *p1, const void *p2)
{
    const int *a = (const int *)p1;
    const int *b = (const int *)p2;
    return a[2] - b[2];
}

// Initialization of parent[] and rank[] arrays
void makeSet(int parent[], int rank[], int n)
{
    for (int i = 0; i < n; i++) {
        parent[i] = i;
        rank[i] = 0;
    }
}

// Function to find the parent of a node
int findParent(int parent[], int component)
{
    if (parent[component] == component)
        return component;

    return parent[component]
        = findParent(parent, parent[component]);
}

// Function to unite two sets
void unionSet(int u, int v, int parent[], int rank[], int n)
{
    u = findParent(parent, u);
    v = findParent(parent, v);

    if (rank[u] < rank[v]) {
        parent[u] = v;
    }
    else if (rank[u] > rank[v]) {
        parent[v] = u;
    }
    else {
        parent[v] = u;
        rank[u]++;
    }
}

// Function to find the MST
int kruskalAlgo(int n, int edge[n][3])
{
    qsort(edge, n, sizeof(edge[0]), comparator);
    int parent[n];
    int rank[n];
    makeSet(parent, rank, n);

    int minCost = 0;
    for (int i = 0; i < n; i++) {
        int v1 = findParent(parent, edge[i][0]);
        int v2 = findParent(parent, edge[i][1]);
        int wt = edge[i][2];

        if (v1 != v2) {
            unionSet(v1, v2, parent, rank, n);
            minCost += wt;
        }
    }

    return minCost;
}

// Driver code
int main()
{
    int edge[5][3] = { { 0, 1, 10 },
                    { 0, 2, 6 },
                    { 0, 3, 5 },
                    { 1, 3, 15 },
                    { 2, 3, 4 } };

    printf("%d\n", kruskalAlgo(5, edge));

    return 0;
}
```

---

## Lab 4: Bellman-Ford Algorithm

```c
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_VERTICES 1000
#define INF INT_MAX

typedef struct {
    int source;
    int destination;
    int weight;
} Edge;

void bellmanFord(int graph[MAX_VERTICES][MAX_VERTICES],
    int vertices, int edges, int source)
{
    int distance[MAX_VERTICES];

    for (int i = 0; i < vertices; ++i)
        distance[i] = INF;
    distance[source] = 0;

    for (int i = 0; i < vertices - 1; ++i) {
        for (int j = 0; j < edges; ++j) {
            if (graph[j][0] != -1
                && distance[graph[j][0]] != INF
                && distance[graph[j][1]]
                > distance[graph[j][0]]
                + graph[j][2])
                distance[graph[j][1]]
                = distance[graph[j][0]] + graph[j][2];
        }
    }

    for (int i = 0; i < edges; ++i) {
        if (graph[i][0] != -1
            && distance[graph[i][0]] != INF
            && distance[graph[i][1]]
            > distance[graph[i][0]] + graph[i][2]) {
            printf("Negative cycle detected\n");
            return;
        }
    }

    printf("Vertex Distance from Source\n");
    for (int i = 0; i < vertices; ++i)
        printf("%d \t\t %d\n", i, distance[i]);
}

int main()
{
    int vertices = 6;
    int edges = 8;

    int graph[MAX_VERTICES][MAX_VERTICES]
        = { { 0, 1, 5 }, { 0, 2, 7 }, { 1, 2, 3 },
            { 1, 3, 4 }, { 1, 4, 6 }, { 3, 4, -1 },
            { 3, 5, 2 }, { 4, 5, -3 } };

    bellmanFord(graph, vertices, edges, 0);
    return 0;
}
```

---

## Lab 5: B-Tree

```cpp
#include <iostream>
using namespace std;

template <typename T, int ORDER>
class BTreeNode {
public:
    T keys[ORDER - 1];
    BTreeNode* children[ORDER];
    int n;
    bool leaf;

    BTreeNode(bool isLeaf = true) : n(0), leaf(isLeaf) {
        for (int i = 0; i < ORDER; i++)
            children[i] = nullptr;
    }
};

template <typename T, int ORDER>
class BTree {
private:
    BTreeNode<T, ORDER>* root;

    void splitChild(BTreeNode<T, ORDER>* x, int i) {
        BTreeNode<T, ORDER>* y = x->children[i];
        BTreeNode<T, ORDER>* z = new BTreeNode<T, ORDER>(y->leaf);
        z->n = ORDER / 2 - 1;

        for (int j = 0; j < ORDER / 2 - 1; j++)
            z->keys[j] = y->keys[j + ORDER / 2];

        if (!y->leaf) {
            for (int j = 0; j < ORDER / 2; j++)
                z->children[j] = y->children[j + ORDER / 2];
        }

        y->n = ORDER / 2 - 1;

        for (int j = x->n; j >= i + 1; j--)
            x->children[j + 1] = x->children[j];

        x->children[i + 1] = z;

        for (int j = x->n - 1; j >= i; j--)
            x->keys[j + 1] = x->keys[j];

        x->keys[i] = y->keys[ORDER / 2 - 1];
        x->n = x->n + 1;
    }

    void insertNonFull(BTreeNode<T, ORDER>* x, T k) {
        int i = x->n - 1;

        if (x->leaf) {
            while (i >= 0 && k < x->keys[i]) {
                x->keys[i + 1] = x->keys[i];
                i--;
            }
            x->keys[i + 1] = k;
            x->n = x->n + 1;
        } else {
            while (i >= 0 && k < x->keys[i])
                i--;

            i++;
            if (x->children[i]->n == ORDER - 1) {
                splitChild(x, i);

                if (k > x->keys[i])
                    i++;
            }
            insertNonFull(x->children[i], k);
        }
    }

    void traverse(BTreeNode<T, ORDER>* x) {
        int i;
        for (i = 0; i < x->n; i++) {
            if (!x->leaf)
                traverse(x->children[i]);
            cout << " " << x->keys[i];
        }

        if (!x->leaf)
            traverse(x->children[i]);
    }

    BTreeNode<T, ORDER>* search(BTreeNode<T, ORDER>* x, T k) {
        int i = 0;
        while (i < x->n && k > x->keys[i])
            i++;

        if (i < x->n && k == x->keys[i])
            return x;

        if (x->leaf)
            return nullptr;

        return search(x->children[i], k);
    }

    T getPredecessor(BTreeNode<T, ORDER>* node, int idx) {
        BTreeNode<T, ORDER>* current = node->children[idx];
        while (!current->leaf)
            current = current->children[current->n];
        return current->keys[current->n - 1];
    }

    T getSuccessor(BTreeNode<T, ORDER>* node, int idx) {
        BTreeNode<T, ORDER>* current = node->children[idx + 1];
        while (!current->leaf)
            current = current->children[0];
        return current->keys[0];
    }

    void fill(BTreeNode<T, ORDER>* node, int idx) {
        if (idx != 0 && node->children[idx - 1]->n >= ORDER / 2)
            borrowFromPrev(node, idx);
        else if (idx != node->n && node->children[idx + 1]->n >= ORDER / 2)
            borrowFromNext(node, idx);
        else {
            if (idx != node->n)
                merge(node, idx);
            else
                merge(node, idx - 1);
        }
    }

    void borrowFromPrev(BTreeNode<T, ORDER>* node, int idx) {
        BTreeNode<T, ORDER>* child = node->children[idx];
        BTreeNode<T, ORDER>* sibling = node->children[idx - 1];

        for (int i = child->n - 1; i >= 0; --i)
            child->keys[i + 1] = child->keys[i];

        if (!child->leaf) {
            for (int i = child->n; i >= 0; --i)
                child->children[i + 1] = child->children[i];
        }

        child->keys[0] = node->keys[idx - 1];

        if (!child->leaf)
            child->children[0] = sibling->children[sibling->n];

        node->keys[idx - 1] = sibling->keys[sibling->n - 1];

        child->n += 1;
        sibling->n -= 1;
    }

    void borrowFromNext(BTreeNode<T, ORDER>* node, int idx) {
        BTreeNode<T, ORDER>* child = node->children[idx];
        BTreeNode<T, ORDER>* sibling = node->children[idx + 1];

        child->keys[child->n] = node->keys[idx];

        if (!child->leaf)
            child->children[child->n + 1] = sibling->children[0];

        node->keys[idx] = sibling->keys[0];

        for (int i = 1; i < sibling->n; ++i)
            sibling->keys[i - 1] = sibling->keys[i];

        if (!sibling->leaf) {
            for (int i = 1; i <= sibling->n; ++i)
                sibling->children[i - 1] = sibling->children[i];
        }

        child->n += 1;
        sibling->n -= 1;
    }

    void merge(BTreeNode<T, ORDER>* node, int idx) {
        BTreeNode<T, ORDER>* child = node->children[idx];
        BTreeNode<T, ORDER>* sibling = node->children[idx + 1];
        child->keys[ORDER / 2 - 1] = node->keys[idx];
        for (int i = 0; i < sibling->n; ++i)
            child->keys[i + ORDER / 2] = sibling->keys[i];
        if (!child->leaf) {
            for (int i = 0; i <= sibling->n; ++i)
                child->children[i + ORDER / 2] = sibling->children[i];
        }
        for (int i = idx + 1; i < node->n; ++i)
            node->keys[i - 1] = node->keys[i];
        for (int i = idx + 2; i <= node->n; ++i)
            node->children[i - 1] = node->children[i];
        child->n += sibling->n + 1;
        node->n--;
        delete sibling;
    }

    void removeFromNonLeaf(BTreeNode<T, ORDER>* node, int idx) {
        T k = node->keys[idx];
        if (node->children[idx]->n >= ORDER / 2) {
            T pred = getPredecessor(node, idx);
            node->keys[idx] = pred;
            remove(node->children[idx], pred);
        } else if (node->children[idx + 1]->n >= ORDER / 2) {
            T succ = getSuccessor(node, idx);
            node->keys[idx] = succ;
            remove(node->children[idx + 1], succ);
        } else {
            merge(node, idx);
            remove(node->children[idx], k);
        }
    }

    void removeFromLeaf(BTreeNode<T, ORDER>* node, int idx) {
        for (int i = idx + 1; i < node->n; ++i)
            node->keys[i - 1] = node->keys[i];
        node->n--;
    }

    void remove(BTreeNode<T, ORDER>* node, T k) {
        int idx = 0;
        while (idx < node->n && node->keys[idx] < k)
            ++idx;
        if (idx < node->n && node->keys[idx] == k) {
            if (node->leaf)
                removeFromLeaf(node, idx);
            else
                removeFromNonLeaf(node, idx);
        } else {
            if (node->leaf) {
                cout << "The key " << k << " is not present in the tree\n";
                return;
            }
            bool flag = ((idx == node->n) ? true : false);
            if (node->children[idx]->n < ORDER / 2)
                fill(node, idx);
            if (flag && idx > node->n)
                remove(node->children[idx - 1], k);
            else
                remove(node->children[idx], k);
        }
    }

public:
    BTree() { root = new BTreeNode<T, ORDER>(true); }
    void insert(T k) {
        if (root->n == ORDER - 1) {
            BTreeNode<T, ORDER>* s = new BTreeNode<T, ORDER>(false);
            s->children[0] = root;
            root = s;
            splitChild(s, 0);
            insertNonFull(s, k);
        } else
            insertNonFull(root, k);
    }

    void traverse() {
        if (root != nullptr)
            traverse(root);
    }

    BTreeNode<T, ORDER>* search(T k) {
        return (root == nullptr) ? nullptr : search(root, k);
    }

    void remove(T k) {
        if (!root) {
            cout << "The tree is empty\n";
            return;
        }

        remove(root, k);

        if (root->n == 0) {
            BTreeNode<T, ORDER>* tmp = root;
            if (root->leaf)
                root = nullptr;
            else
                root = root->children[0];
            delete tmp;
        }
    }
};

int main() {
    BTree<int, 3> t;

    t.insert(10);
    t.insert(20);
    t.insert(5);
    t.insert(6);
    t.insert(12);
    t.insert(30);
    t.insert(7);
    t.insert(17);

    cout << "Traversal of the constructed tree is: ";
    t.traverse();
    cout << endl;

    int k = 6;
    (t.search(k) != nullptr) ? cout << k << " is found" << endl
        : cout << k << " is not found" << endl;

    k = 15;
    (t.search(k) != nullptr) ? cout << k << " is found" << endl
        : cout << k << " is not found" << endl;

    t.remove(6);
    cout << "Traversal of the tree after removing 6: ";
    t.traverse();
    cout << endl;

    t.remove(13);
    cout << "Traversal of the tree after removing 13: ";
    t.traverse();
    cout << endl;

    return 0;
}
```

---

## Lab 6: KMP String Matching

```cpp
#include <iostream>
#include <string>
#include <vector>
using namespace std;

void constructLps(string &pat, vector<int> &lps) {
    int len = 0;
    lps[0] = 0;
    int i = 1;
    while (i < pat.length()) {
        if (pat[i] == pat[len]) {
            len++;
            lps[i] = len;
            i++;
        }
        else {
            if (len != 0) {
                len = lps[len - 1];
            }
            else {
                lps[i] = 0;
                i++;
            }
        }
    }
}

vector<int> search(string &pat, string &txt) {
    int n = txt.length();
    int m = pat.length();

    vector<int> lps(m);
    vector<int> res;

    constructLps(pat, lps);

    int i = 0;
    int j = 0;

    while (i < n) {
        if (txt[i] == pat[j]) {
            i++;
            j++;
        }
        if (j == m) {
            res.push_back(i - j);
            j = lps[j - 1];
        }
        else {
            if (j != 0)
                j = lps[j - 1];
            else
                i++;
        }
    }
    return res;
}

int main() {
    string txt = "aabaacaadaabaaba";
    string pat = "aaba";

    vector<int> res = search(pat, txt);
    for (int i = 0; i < res.size(); i++)
        cout << res[i] << " ";

    return 0;
}
```

---

## Lab 7: Suffix Tree

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define MAX_CHAR 256

struct SuffixTreeNode {
    struct SuffixTreeNode *children[MAX_CHAR];
    struct SuffixTreeNode *suffixLink;
    int start;
    int *end;
    int suffixIndex;
};

typedef struct SuffixTreeNode Node;

char text[100];
Node *root = NULL;

Node *lastNewNode = NULL;
Node *activeNode = NULL;

int activeEdge = -1;
int activeLength = 0;

int remainingSuffixCount = 0;
int leafEnd = -1;
int *rootEnd = NULL;
int *splitEnd = NULL;
int size = -1;

Node *newNode(int start, int *end)
{
    Node *node =(Node*) malloc(sizeof(Node));
    int i;
    for (i = 0; i < MAX_CHAR; i++)
        node->children[i] = NULL;

    node->suffixLink = root;
    node->start = start;
    node->end = end;
    node->suffixIndex = -1;
    return node;
}

int edgeLength(Node *n) {
    if(n == root)
        return 0;
    return *(n->end) - (n->start) + 1;
}

int walkDown(Node *currNode)
{
    if (activeLength >= edgeLength(currNode))
    {
        activeEdge += edgeLength(currNode);
        activeLength -= edgeLength(currNode);
        activeNode = currNode;
        return 1;
    }
    return 0;
}

void extendSuffixTree(int pos)
{
    leafEnd = pos;
    remainingSuffixCount++;
    lastNewNode = NULL;

    while(remainingSuffixCount > 0) {
        if (activeLength == 0)
            activeEdge = pos;

        if (activeNode->children[text[activeEdge]] == NULL)
        {
            activeNode->children[text[activeEdge]] =
                newNode(pos, &leafEnd);

            if (lastNewNode != NULL)
            {
                lastNewNode->suffixLink = activeNode;
                lastNewNode = NULL;
            }
        }
        else
        {
            Node *next = activeNode->children[text[activeEdge]];
            if (walkDown(next))
            {
                continue;
            }
            if (text[next->start + activeLength] == text[pos])
            {
                if(lastNewNode != NULL && activeNode != root)
                {
                    lastNewNode->suffixLink = activeNode;
                    lastNewNode = NULL;
                }

                activeLength++;
                break;
            }

            splitEnd = (int*) malloc(sizeof(int));
            *splitEnd = next->start + activeLength - 1;

            Node *split = newNode(next->start, splitEnd);
            activeNode->children[text[activeEdge]] = split;

            split->children[text[pos]] = newNode(pos, &leafEnd);
            next->start += activeLength;
            split->children[text[next->start]] = next;

            if (lastNewNode != NULL)
            {
                lastNewNode->suffixLink = split;
            }

            lastNewNode = split;
        }

        remainingSuffixCount--;
        if (activeNode == root && activeLength > 0)
        {
            activeLength--;
            activeEdge = pos - remainingSuffixCount + 1;
        }
        else if (activeNode != root)
        {
            activeNode = activeNode->suffixLink;
        }
    }
}

void print(int i, int j)
{
    int k;
    for (k=i; k<=j && text[k] != '#'; k++)
        printf("%c", text[k]);
    if(k<=j)
        printf("#");
}

void setSuffixIndexByDFS(Node *n, int labelHeight)
{
    if (n == NULL) return;

    if (n->start != -1)
    {
        print(n->start, *(n->end));
    }
    int leaf = 1;
    int i;
    for (i = 0; i < MAX_CHAR; i++)
    {
        if (n->children[i] != NULL)
        {
            if (leaf == 1 && n->start != -1)
                printf("%d]\n", n->suffixIndex);

            leaf = 0;
            setSuffixIndexByDFS(n->children[i], labelHeight + edgeLength(n->children[i]));
        }
    }
    if (leaf == 1)
    {
        for (i = n->start; i<= *(n->end); i++)
        {
            if (text[i] == '#')
            {
                n->end = (int*) malloc(sizeof(int));
                *(n->end) = i;
            }
        }
        n->suffixIndex = size - labelHeight;
        printf("%d\n", n->suffixIndex);
    }
}

void freeSuffixTreeByPostOrder(Node *n)
{
    if (n == NULL)
        return;
    int i;
    for (i = 0; i < MAX_CHAR; i++)
    {
        if (n->children[i] != NULL)
        {
            freeSuffixTreeByPostOrder(n->children[i]);
        }
    }
    if (n->suffixIndex == -1)
        free(n->end);
    free(n);
}

void buildSuffixTree()
{
    size = strlen(text);
    int i;
    rootEnd = (int*) malloc(sizeof(int));
    *rootEnd = -1;

    root = newNode(-1, rootEnd);

    activeNode = root;
    for (i=0; i<size; i++)
        extendSuffixTree(i);
    int labelHeight = 0;
    setSuffixIndexByDFS(root, labelHeight);

    freeSuffixTreeByPostOrder(root);
}

int main(int argc, char *argv[])
{
    strcpy(text, "xabxa#babxba$"); buildSuffixTree();
    return 0;
}
```
