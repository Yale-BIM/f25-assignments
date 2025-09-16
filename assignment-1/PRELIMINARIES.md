## Preliminaries

### Notation
We refer to `vectors` or column matrices with bold lower-case letters (e.g., ![equation](https://latex.codecogs.com/png.latex?%5Cbold%7Bx%7D) <!--$`\bold{x}`$-->).
Other `matrices`, such as linear transformations, and `scalars` are written with regular
font weight. 

### Conventions

In this course, all reasoning in space is done in a 
[right hand system](http://mathworld.wolfram.com/Right-HandRule.html). The orientation
of the cross product between ![equation](https://latex.codecogs.com/png.latex?%5Cbold%7Bi%7D%20%3D%20%5B1%2C%200%2C%200%5D%5ET) <!--$`\bold{i} = [1, 0, 0]^T`$--> (in the direction of the ![equation](https://latex.codecogs.com/png.latex?x) <!--$`x`$-->axis) and 
![equation](https://latex.codecogs.com/png.latex?%5Cbold%7Bj%7D%20%3D%20%5B0%2C%201%2C%200%5D%5ET)<!--$`\bold{j} = [0, 1, 0]^T`$--> (![equation](https://latex.codecogs.com/png.latex?y)<!--$`y`$--> axis) is determined by
placing ![equation](https://latex.codecogs.com/png.latex?%5Cbold%7Bi%7D)<!--$`\bold{i}`$--> and ![equation](https://latex.codecogs.com/png.latex?%5Cbold%7Bj%7D)<!--$`\bold{j}`$--> tail-to-tail, flattening the right hand, extending it in the direction
of ![equation](https://latex.codecogs.com/png.latex?%5Cbold%7Bi%7D)<!--$`\bold{i}`$-->, and then curling the fingers towards ![equation](https://latex.codecogs.com/png.latex?%5Cbold%7Bj%7D)<!--$`\bold{j}`$-->. The thumb then points in the direction
of ![equation](https://latex.codecogs.com/png.latex?%5Cbold%7Bk%7D%20%3D%20%5Cbold%7Bi%7D%20%5Ctimes%20%5Cbold%7Bj%7D%20%3D%20%5B0%2C0%2C1%5D%5ET)<!--$`\bold{k} = \bold{i} \times \bold{j} = [0,0,1]^T`$--> (corresponding to the ![equation](https://latex.codecogs.com/png.latex?z)<!--$`z`$--> axis). 

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/d/d2/Right_hand_rule_cross_product.svg"
width="100" alt="Right hand system from Wikipedia.org"/><br>
Right Hand System (image from Wikipedia.org)
</p>

We also use a [right hand rule](https://en.wikipedia.org/wiki/Right-hand_rule#Rotations) 
for rotations: right fingers are curled in the direction of rotation and the right thumb 
points in the positive direction of the axis.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Right-hand_grip_rule.svg/220px-Right-hand_grip_rule.svg.png"
width="75" alt="Right hand rule from Wikipedia.org"/><br>
Right Hand Rule (image from Wikipedia.org)
</p>

### Kinematic Chains

A kinematic chain is an assembly of rigid bodies or `links` connected by `joints`.
The joints allow the links to move relative to one another
and are typically instrumented with sensors, e.g., to measure the relative position of neighboring links.

There are different types of joints but, in this assignment, we will focus on
working with `revolute joints` because Shutter has 4 of them. Revolute joints
have a single axis of rotation and, thus, exibit just one Degree of Freedom. The 
`joint angle` of these revolute joints controls the displacement between the pair
of links that are connected to it.

<p align="center">
<img src="https://circuitdigest.com/sites/default/files/inlineimages/Revolute-Joint.gif" width="150" alt="Revolute joint from circuitdigest.com"/><br/>
Revolute Joint (image from circuitdigest.com)
</p>

In general, we think about Degrees of Freedom (DoF) as the number of independent variables 
that need to be specified in order to locate all parts of a robot.
Shutter, has 4 servos in its arm, each of which implements a revolute joint. Thus, Shutter has 4 DoF. 


### 3D Transformations

3D spatial transformations map 3D points from one `coordinate system` (or `frame`) to another.
They are particularly relevant for robotics and 3D vision applications, where the 
elements of interest are in different locations in the world. For example, transformations
are useful to know the position of the camera in Shutter relative to one of its 
links or its base. Similarly, 3D transformations can help infer the location of an object 
with respect to a camera that observes it.

Following [ROS conventions](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Tf2-Main.html), 
we refer to a point ![equation](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bp%7D)<!--$`\mathbf{p}`$--> within a frame ![equation](https://latex.codecogs.com/png.latex?B)<!--$`B`$--> as ![equation](https://latex.codecogs.com/png.latex?%5E%7BB%7D%5Cmathbf%7Bp%7D)<!--$`^{B}\mathbf{p}`$-->. 
We also refer to the relationship between any two frames ![equation](https://latex.codecogs.com/png.latex?A)<!--$`A`$--> and ![equation](https://latex.codecogs.com/png.latex?B)<!--$`B`$--> as 
a 6 Degrees of Freedom (DoF) transformation: 
a rotation followed by a translation. Specifically,
the pose of ![equation](https://latex.codecogs.com/png.latex?A)<!--$`A`$--> in ![equation](https://latex.codecogs.com/png.latex?B)<!--$`B`$--> is given by the rotation of ![equation](https://latex.codecogs.com/png.latex?A)<!--$`A`$-->'s coordinate axes in ![equation](https://latex.codecogs.com/png.latex?B)<!--$`B`$-->
 and the translation from ![equation](https://latex.codecogs.com/png.latex?B)<!--$`B`$-->'s origin to ![equation](https://latex.codecogs.com/png.latex?A)<!--$`A`$-->'s origin. 

>- **Translations:** A 3D translation can be represented by a vector ![equation](https://latex.codecogs.com/png.latex?%5Cbold%7Bt%7D%20%3D%20%5Bt_1%2C%20t_2%2C%20t_3%5D)<!--$`\bold{t} = [t_1, t_2, t_3]`$-->
    or by a ![equation](https://latex.codecogs.com/png.latex?4%20%5Ctimes%204)<!--$`4 \times 4`$--> matrix:<br>
    ![equation](https://latex.codecogs.com/png.latex?t%20%3D%20%5Cbegin%7Bbmatrix%7D%201%20%26%200%20%26%200%20%26%20t_1%5C%5C%200%20%26%201%20%26%200%20%26%20t_2%5C%5C%200%20%26%200%20%26%201%20%26%20t_3%5C%5C%200%20%26%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7D)<!--$`t =
    \begin{bmatrix}
    1 & 0 & 0 & t_1\\
    0 & 1 & 0 & t_2\\
    0 & 0 & 1 & t_3\\
    0 & 0 & 0 & 1
    \end{bmatrix}`$--><br>
    The scalars ![equation](https://latex.codecogs.com/png.latex?t_1)<!--$`t_1`$-->, ![equation](https://latex.codecogs.com/png.latex?t_2)<!--$`t_2`$-->, and ![equation](https://latex.codecogs.com/png.latex?t_3)<!--$`t_3`$--> correspond to the displacements in ![equation](https://latex.codecogs.com/png.latex?x)<!--$`x`$-->,
    ![equation](https://latex.codecogs.com/png.latex?y)<!--$`y`$-->, and ![equation](https://latex.codecogs.com/png.latex?z)<!--$`z`$-->, respectively. Thus, a translation has 3 DoF. Note that
    representing translations with ![equation](https://latex.codecogs.com/png.latex?4%20%5Ctimes%204)<!--$`4 \times 4`$--> matrices as above is helpful 
    for transforming points in homogeneous coordinates.<br>
>- **Rotations:** A 3D rotation has 3 DoF as well. Each DoF corresponds to a rotation around one of the axes of the 
    coordinate frame. We can represent rotations also as ![equation](https://latex.codecogs.com/png.latex?4%20%5Ctimes%204)<!--$`4 \times 4`$--> transformation matrices:<br>
    ![equation](https://latex.codecogs.com/png.latex?R%20%3D%20%5Cbegin%7Bbmatrix%7D%20r_%7B11%7D%20%26%20r_%7B12%7D%20%26%20r_%7B13%7D%20%26%200%5C%5C%20r_%7B21%7D%20%26%20r_%7B22%7D%20%26%20r_%7B23%7D%20%26%200%5C%5C%20r_%7B31%7D%20%26%20r_%7B32%7D%20%26%20r_%7B33%7D%20%26%200%5C%5C%200%20%26%200%20%26%200%20%26%201%20%5Cend%7Bbmatrix%7D)
    <!--$`R = 
    \begin{bmatrix}
    r_{11} & r_{12} & r_{13} & 0\\
    r_{21} & r_{22} & r_{23} & 0\\
    r_{31} & r_{32} & r_{33} & 0\\
    0 & 0 & 0 & 1
    \end{bmatrix}`$-->
    <br>
    Note that the ![equation](https://latex.codecogs.com/png.latex?3%20%5Ctimes%203) <!--$`3 \times 3`$--> submatrix of ![equation](https://latex.codecogs.com/png.latex?R)<!--$`R`$--> with the elements ![equation](https://latex.codecogs.com/png.latex?r_%7B11%7D%20%5Cldots%20r_%7B33%7D) <!--$`r_{11}`$ ... $`r_{33}`$-->
    is an [orthogonal matrix](https://en.wikipedia.org/wiki/Orthogonal_matrix).<br>    
    It is important to know that [ROS uses quaternions](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Learning-About-Tf2-And-Time-Cpp.html) 
    to represent rotations, but there are many other useful representations (e.g., 
    [Euler angles](https://en.wikipedia.org/wiki/Euler_angles)).

### Changing the Frame of a Point
Let ![equation](https://latex.codecogs.com/png.latex?%5E%7BA%7D%5Cmathbf%7Bp%7D)<!--$`^{A}\mathbf{p}`$--> be a 3D point in the ![equation](https://latex.codecogs.com/png.latex?A)<!--$`A`$--> frame. Its position in 
![equation](https://latex.codecogs.com/png.latex?B)<!--$`B`$--> can be expressed as ![equation](https://latex.codecogs.com/png.latex?%5E%7BB%7D%5Cmathbf%7Bp%7D%20%3D%20%5E%7BB%7D_%7BA%7DT%5C%20%5E%7BA%7D%5Cmathbf%7Bp%7D)<!--$`^{B}\mathbf{p} = ^{B}_{A}T\ ^{A}\mathbf{p}`$-->, where 
![equation](https://latex.codecogs.com/png.latex?%5E%7BB%7D_%7BA%7DT%20%3D%20%5E%7BB%7D_%7BA%7D%28t%20%5Ctimes%20R%29)<!--$`^{B}_{A}T = ^{B}_{A}(t \times R)`$--> is the ![equation](https://latex.codecogs.com/png.latex?4%20%5Ctimes%204)<!--$`4 \times 4`$--> transformation matrix that
results from right-multiplying the translation matrix ![equation](https://latex.codecogs.com/png.latex?%5E%7BB%7D_%7BA%7Dt)<!--$`^{B}_{A}t`$--> by the rotation
matrix ![equation](https://latex.codecogs.com/png.latex?%5E%7BB%7D_%7BA%7DR)<!--$`^{B}_{A}R`$-->. In particular,
 
- ![equation](https://latex.codecogs.com/png.latex?%5E%7BB%7D_%7BA%7Dt)<!--$`^{B}_{A}t`$--> is the ![equation](https://latex.codecogs.com/png.latex?4%20%5Ctimes%204)<!--$`4 \times 4`$--> transformation matrix that encodes the
 translation between the frames ![equation](https://latex.codecogs.com/png.latex?A)<!--$`A`$--> and ![equation](https://latex.codecogs.com/png.latex?B)<!--$`B`$-->. The values ![equation](https://latex.codecogs.com/png.latex?t_1%2C%20t_2%2C%20t_3)<!--$`t_1, t_2, t_3`$--> of
the translation ![equation](https://latex.codecogs.com/png.latex?%5E%7BB%7D_%7BA%7Dt)<!--$`^{B}_{A}t`$--> are the origin of the frame ![equation](https://latex.codecogs.com/png.latex?A)<!--$`A`$--> in ![equation](https://latex.codecogs.com/png.latex?B)<!--$`B`$-->.
- ![equation](https://latex.codecogs.com/png.latex?%5E%7BB%7D_%7BA%7DR)<!--$`^{B}_{A}R`$--> is the  ![equation](https://latex.codecogs.com/png.latex?4%20%5Ctimes%204)<!--$`4 \times 4`$--> rotation matrix corresponding to the orientation of ![equation](https://latex.codecogs.com/png.latex?A)<!--$`A`$-->'s coordinate axes in 
![equation](https://latex.codecogs.com/png.latex?B)<!--$`B`$-->. 

Note that the 3D vector with elements ![equation](https://latex.codecogs.com/png.latex?r_%7B11%7D%2C%20r_%7B21%7D%2C%20r_%7B31%7D)<!--$`r_{11}, r_{21}, r_{31}`$--> 
from the first column of the rotation matrix ![equation](https://latex.codecogs.com/png.latex?%5E%7BB%7D_%7BA%7DR)<!--$`^{B}_{A}R`$--> has the same direction as the ![equation](https://latex.codecogs.com/png.latex?x)<!--$`x`$--> axis of ![equation](https://latex.codecogs.com/png.latex?A)<!--$`A`$--> 
in the ![equation](https://latex.codecogs.com/png.latex?B)<!--$`B`$--> frame. Similarly, the elements ![equation](https://latex.codecogs.com/png.latex?r_%7B12%7D%2C%20r_%7B22%7D%2C%20r_%7B32%7D)<!--$`r_{12}, r_{22}, r_{32}`$-->
and ![equation](https://latex.codecogs.com/png.latex?r_%7B13%7D%2C%20r_%7B23%7D%2C%20r_%7B33%7D)<!--$`r_{13}, r_{23}, r_{33}`$--> have the same direction of the ![equation](https://latex.codecogs.com/png.latex?y)<!--$`y`$--> and ![equation](https://latex.codecogs.com/png.latex?z)<!--$`z`$--> axes of 
![equation](https://latex.codecogs.com/png.latex?A)<!--$`A`$--> in ![equation](https://latex.codecogs.com/png.latex?B)<!--$`B`$-->, respectively.

### Transforms in ROS 2

The [tf2](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Tf2-Main.html) library in ROS 2 represents transforms and coordinate frames 
in a `tree structure` buffered in time. The tree is a directed graph with a root. Any two 
vertices in it are connected by one path. The nodes of the graph correspond to coordinate frames,
each associated with a link, and the edges correspond to transforms between pairs of frames. 

<p align="center">
<img src="https://docs.ros.org/en/jazzy/_images/turtlesim_frames.png" width="400" alt="Example tf2 tree from https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Tf2/Introduction-To-Tf2.html"/><br/>
Example tf2 tree (image from https://docs.ros.org/en/jazzy/Tutorials)
</p>

Any directed edge in the tf2 tree has a `parent` frame (source node), and a `child` frame 
(target node). Let the parent frame be ![equation](https://latex.codecogs.com/png.latex?P)<!--$`P`$--> and the child be ![equation](https://latex.codecogs.com/png.latex?C)<!--$`C`$-->. Then, the transform
stored in the edge parent -> child corresponds to ![equation](https://latex.codecogs.com/png.latex?%5E%7BP%7D_%7BC%7DT)<!--$`^{P}_{C}T`$-->.

The tf2 library quickly computes the net transform between two nodes (frames) 
by multiplying the edges connecting them. To traverse up a directed edge from a child to a parent node, 
tf2 uses the inverse of the transformation that is stored in the edge.

Now that you are done reading these notes, go back to working on [assignment 1](README.md).