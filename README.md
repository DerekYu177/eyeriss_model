# TLM for Eyeriss as a Systolic Array
This is a Transactional Level Model (TLM) for a modification on Eyeriss, a low-power hardware accelerator for machine learning.
Eyeriss manages it's impressive low power consumption by exploiting data reuse across the system, as opposed to optimizing only one portion of the model.

## "Modifications on Eyeriss"
The model presented here is not a cycle accurate model, nor is it an accurate TLM of Eyeriss itself.
This model is specifically a TLM of Eyeriss as a Systolic Array, where a grid of Processing Elements (PEs) are essentially in lockstep as they compute convolutions.

## How it works
At the top level, the accelerator generates a grid of PEs. Once given an input feature map (ifmap), and a kernel, it will then compute the convolutional output feature map (ofmap).
This is done by first feeding in the ifmap in a diagonal pattern along the bottom and left hand edges of the PE array.
The kernels are fed in along the left hand edge.
At each delta step, PEs that have both a valid ifmap row and kernel row compute a partial sum (psum).
The ifmaps move in a diagonal direction towards the top right hand corner, the ifmaps move towards the right, and the psums move upwards.
We enforce the accumulation of the psums in the correct order.
At the end of the step, the accumulation of the psums at the top of the PE array are the ofmap rows.
These are then stitched together to form the ofmap.

## What is special about this model
We build a cost tracker into every processing element.

### Memory Cost
Every transaction has a memory cost associated with it.
Eyeriss has four memory layers: DRAM, Global Buffer, Inter-PE, and SPAD.
The last isn't exact, technically each PE has three smaller distinct SPADs that we have crudely combined together.
Our model does not support Global Buffer due to programmer availability.
If you would like to see this happen, email me or create an issue and we can chat about it there.

### Computational Cost
Each computation also has a cost to it, which we record in the total number of additions and multiplications.
I have not yet thought of any other metric that we need to record in this category.
Again, if you want to see something else happen, let's discuss.

## Model Limitations
Due to my freedom as a graduate student, I have not been able to complete some aspects of this project.
1. Support for a stride greater than 1x1: I have a test that does the computation for a 7x7, which was our ultimate goal. It fails right now. I have made some progress, namely each row is computed correctly, but a stride length of greater than x1 means that rows need to be compacted; I do not have this working yet.
2. Global Cost Tracker on an Accelerator level: I have implemented this in terms of looking at the cost for each PE and then summing them together. If you want better reporting, that class needs to be finished.
3. Support for multiple ifmaps: This shouldn't be too hard, since we can only compute one after the output of an ofmap. Unless I am missing something.
4. Mapping fully connected layers onto PE: If you're from the eyeriss team, I would love to pick your brain.

## Operation
`runnable.py` contains all of the code necessary to run this project and produce an output `output_filled.xml`.
If you opened the file, you would understand the necessity of a global cost tracker.

## Example

Suppose you had a PE array of 2x2:
+__-+-+
|B|D|
+_-+_-+
|A|C|
+_-+_-+

An ifmap of size 5x5:
[
  [1 ...  5],
  [6 ...  10],
  [11 ... 15],
  [16 ... 20],
  [21 ... 25],
]

and a kernel of size 2x2:
[
  [1, 2],
  [3, 4]
]

and a stride of 1x1.

### Step 1
At the first step, the kernels are inserted into the leftmost side:
+__------------------+----------------+
| B(kernel=[1, 2]) | D(kernel=None) |
+-----------------+-----------------+
| A(kernel=[3, 4]) | C(kernel=None) |
+-----------------+-----------------+

Then the ifmaps are inserted along the bottom and leftmost side:
+__----------------------------------+----------------------------+
| B(kernel=[1, 2], ifmap=[1...5])  | D(kernel=None, ifmap=None) |
+----------------------------------+----------------------------+
| A(kernel=[3, 4], ifmap=[6...10]) | C(kernel=None, ifmap=None) |
+----------------------------------+----------------------------+

The convolution is computed on a per PE basis, starting from the bottom half:

#### Delta Step
+__---------------------------------------------------------+---------------------------------------+
| B(kernel=[1, 2], ifmap=[1...5], psum=None)              | D(kernel=None, ifmap=None, psum=None) |
+---------------------------------------------------------+---------------------------------------+
| A(kernel=[3, 4], ifmap=[6...10], psum=[46, 53, 60, 67]) | C(kernel=None, ifmap=None, psum=None) |
+---------------------------------------------------------+---------------------------------------+
#### Delta Step 2
The psum from the A is sent to B, and B computes the following accumulation psum = psum + conv(kernel, ifmap)
+__---------------------------------------------------------+---------------------------------------+
| B(kernel=[1, 2], ifmap=[1...5], psum=[51, 61, 71, 81])  | D(kernel=None, ifmap=None, psum=None) |
+---------------------------------------------------------+---------------------------------------+
| A(kernel=[3, 4], ifmap=[6...10], psum=[46, 53, 60, 67]) | C(kernel=None, ifmap=None, psum=None) |
+---------------------------------------------------------+---------------------------------------+
#### Delta Step 3
The final psum at B (the topmost of the PE array) is inserted into the ofmap (4x4)
Ofmap = [
  [51, 61, 71, 81],
  None,
  None,
  None,
]

### Step 2
The ifmaps are moved diagonally upwards; the amount that they move by is dependent on the stride.
A stride of 1x1 will move the ifmap from A to D.
The ifmaps from B, C will move off the PE array, and they will be discarded.
We reset all of the psums.
New ifmaps will be inserted into the PE array.
Kernels are propagated to the right.
Since we have not finished the ofmap, we keep the kernel the same

+__-----------------------------------------------+----------------------------------------------+
| B(kernel=[1, 2], ifmap=[16...20], psum=None)  | D(kernel=[1, 2], ifmap=[6...10], psum=None)  |
+-----------------------------------------------+----------------------------------------------+
| A(kernel=[3, 4], ifmap=[21...25], psum=None)  | C(kernel=[3, 4], ifmap=[11...15], psum=None) |
+-----------------------------------------------+----------------------------------------------+

And now we repeat the delta steps, produce ofmap row 2, and continue.

## Repo
Since this is a pretty barebones project, I have no included any requirements.txt
