# # Computational Mathematics
# ## Project A: Convex Hull
# ## 2025-4-25

import matplotlib.pyplot as plt
import math
import random

#
# ***
# ### Question A.1.
#


# The convex hull of a set of discrete n points in 1D (a line) is exactly $[min point, max point]$.

# Algorithm: Traverse the given point set to find minimum and maximum and return $[min, max]$.

#Runtime complexity: Traverse the point set exactly once, so $O(n)$. 


#
# ***
# ### Question A.2.
#


# Consider apply convex hull algorithm to $\{(n_i, n_i^2),i=1,2,...,n\}$: Note that
# (i) Non-linearity avoids degenerating into the 1D case. (ii) The mapped shape is itself convex.

#Algorithm:

#1. Apply our convex hull algorithm on the points $(n_i, n_i^2)$, $i=1,2,...,N$.

#2. The x-components of the convex hull is the set S sorted in cyclic order.

#Convex hull algorithm can work as sorting algorithm, so its complexity shouldn't be smaller than the lowerbound of sorting.
#Hence the minimal runtime complexity of convex hull algorithm in 2D is at least $O(nlogn)$.


#
# ***
# ### Question A.3.
#

def cross(x, y, c):
    #cross>0: left turn, cross<0: right turn, cross=0: colinear
    return (y[0]-x[0])*(c[1]-x[1]) - (y[1]-x[1])*(c[0]-x[0])

def distance_squared(x, y):
    return (x[0]-y[0])**2 + (x[1]-y[1])**2
def polar_angle(x, y):
    return math.atan2(x[1]-y[1], x[0]-y[0])

def Graham_scan(points):
    pts = points.copy()
    if len(pts) <= 2:
        return pts
    
    b = min(pts, key = lambda pt:(pt[1], pt[0]))
    
    pts.sort(key =lambda p:(polar_angle(p,b), distance_squared(p,b)))
    
    vertices = [pts[0], pts[1]]

    for c in pts[2:]:
        #len >= 2 is to take the case that b a_1 a_2 a_3... is colinear into consideration
        while len(vertices) >= 2 and cross(vertices[-2], vertices[-1], c) <= 0:
            vertices.pop()
        vertices.append(c)

    return vertices

def plotting(pts, vertices):
    x,y = zip(*pts)
    vx, vy = zip(*(vertices + [vertices[0]]))
    plt.scatter(x, y, c='black', label='Given points')
    plt.plot(vx, vy, c='blue', label='Convex hull')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Convex Hull')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

pts = [(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5), (0.25, 0), (1, 0.25), (0.75, 1), (0, 0.75)]
plotting(pts, Graham_scan(pts))


#
# ***
# ### Question A.4.
#

for i in range(10):
    pts = [(random.uniform(0,1), random.uniform(0,1)) for _ in range(50)]
    print(f"Test {i+1}:")
    print("Points:", pts)
    print("Convex Hull:", Graham_scan(pts), "\n")
    plotting(pts, Graham_scan(pts))

#
# ***
# ### Question A.5.
#


# Complexity Analysis

#1. Finding the pivot has complexity $O(n)$, as it traverses the point set only once (although with multiple parameters, but this just contributes to a constant to complexity).

#2. Sorting points has the same complexity as the sort function, which is complexity $O(nlogn)$.

#3. Let's examine what does the scan do to a fixed point in forward-traversal and backtracking. Firstly it will be visited once in the forward traversal.
# The next backtracking might pop it out or incorporate it into the hull. Once incorporated it won't be back-visited anymore.
# Hence a given point will be visited once in the forward traversal and only once in all the backtrackings. The complexity is $O(n)$.

#The total complexity is $O(nlogn)$.


#
# ***
# ### Question A.6.
#


def lower_tangent(A, B):
    ia = A.index(max(A, key=lambda p: (p[0], -p[1])))
    ib = B.index(min(B, key=lambda p: (p[0], p[1])))
    
    while True:
        update = False
        
        original_ib = ib
        while (ib + 1) % len(B) != original_ib:
            c = cross(A[ia], B[ib], B[(ib + 1) % len(B)])
            if c < 0 or (c == 0 and distance_squared(A[ia], B[(ib + 1) % len(B)]) > distance_squared(A[ia], B[ib])):
                ib = (ib + 1) % len(B)
                update = True
            else: break

        original_ia = ia
        while (ia - 1) % len(A) != original_ia:
            c = cross(B[ib], A[ia], A[(ia - 1) % len(A)])
            if c > 0 or (c == 0 and distance_squared(B[ib], A[(ia - 1) % len(A)]) > distance_squared(B[ib], A[ia])):
                ia = (ia - 1) % len(A)
                update = True
            else: break

        if not update:
            break

    return (ia, ib)

def upper_tangent(A, B):
    A_ = [(p[0], -p[1]) for p in A][::-1]
    B_ = [(p[0], -p[1]) for p in B][::-1]
    ia_, ib_ = lower_tangent(A_, B_)

    return (len(A) - 1 - ia_, len(B) - 1 - ib_)

def merge(A, B):
    if A == []: return B
    if B == []: return A

    ia_u, ib_u = upper_tangent(A, B)
    ia_l, ib_l = lower_tangent(A, B)

    hull = []

    i = ia_u
    while hull.append(A[i]) or i != ia_l:
        i = (i + 1) % len(A)
    
    i = ib_l
    while hull.append(B[i]) or i != ib_u:
        i = (i + 1) % len(B)

    return hull

def divide_and_conquer(points):
    if len(points) <= 5:
        return Graham_scan(points)
    pts = sorted(points, key=lambda p: (p[0],p[1]))
    
    midval = len(pts)//2
    A = divide_and_conquer(pts[:midval])
    B = divide_and_conquer(pts[midval:])
    
    return merge(A, B)

#Test this algorithm
pts = [(i, 0) for i in range(10)] + [(i, 1) for i in range(10)] + [(i, 2) for i in range(10)]
plotting(pts, divide_and_conquer(pts))

#
# ***
# ### Question A.7.
#

for i in range(10):
    pts = [(random.uniform(0,1), random.uniform(0,1)) for _ in range(50)]
    print(f"Test {i+1}:")
    print("Points:", pts)
    print("Convex Hull:", Graham_scan(pts), "\n")
    plotting(pts, divide_and_conquer(pts))

#
# ***
# ### Question A.8.
#

#For a convex hull in 3D, each of its face is a polygon and must has at least three edges.
# Each edge is shared by exactly two faces, so we have $2E \geq 3F$.
# $F=2+E-V$ so $2E \geq 6+3E-3V$, thus 
# $$ E \leq 3V-6 < 3V $$.
# $$F=2+E-V \leq 2V-4<2V$$.