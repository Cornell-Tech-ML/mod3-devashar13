# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```


## Paralell check 

```
❯ python project/parallel_check.py
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (163)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (163) 
-------------------------------------------------------------------------|loop #ID
    def _map(                                                            | 
        out: Storage,                                                    | 
        out_shape: Shape,                                                | 
        out_strides: Strides,                                            | 
        in_storage: Storage,                                             | 
        in_shape: Shape,                                                 | 
        in_strides: Strides,                                             | 
    ) -> None:                                                           | 
        # TODO: Implement for Task 3.1.                                  | 
        # raise NotImplementedError("Need to implement for Task 3.1")    | 
        for out_idx in prange(len(out)):---------------------------------| #2
            # Local buffers for thread safety                            | 
            out_midx = np.zeros(MAX_DIMS, np.int32)----------------------| #0
            in_midx = np.zeros(MAX_DIMS, np.int32)-----------------------| #1
                                                                         | 
            # Compute indices                                            | 
            to_index(out_idx, out_shape, out_midx)                       | 
            broadcast_index(out_midx, out_shape, in_shape, in_midx)      | 
                                                                         | 
            # Map function                                               | 
            in_idx = index_to_position(in_midx, in_strides)              | 
            out_pos = index_to_position(out_midx, out_strides)           | 
            out[out_pos] = fn(in_storage[in_idx])                        | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--2 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--2 (parallel)
   +--0 (serial)
   +--1 (serial)


 
Parallel region 0 (loop #2) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#2).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (175) is 
hoisted out of the parallel loop labelled #2 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_midx = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (176) is 
hoisted out of the parallel loop labelled #2 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: in_midx = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (213)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (213) 
---------------------------------------------------------------------|loop #ID
    def _zip(                                                        | 
        out: Storage,                                                | 
        out_shape: Shape,                                            | 
        out_strides: Strides,                                        | 
        a_storage: Storage,                                          | 
        a_shape: Shape,                                              | 
        a_strides: Strides,                                          | 
        b_storage: Storage,                                          | 
        b_shape: Shape,                                              | 
        b_strides: Strides,                                          | 
    ) -> None:                                                       | 
        for out_idx in prange(len(out)):-----------------------------| #6
            # Local buffers for thread safety                        | 
            out_midx = np.zeros(MAX_DIMS, np.int32)------------------| #3
            a_midx = np.zeros(MAX_DIMS, np.int32)--------------------| #4
            b_midx = np.zeros(MAX_DIMS, np.int32)--------------------| #5
                                                                     | 
            # Compute indices                                        | 
            to_index(out_idx, out_shape, out_midx)                   | 
            broadcast_index(out_midx, out_shape, a_shape, a_midx)    | 
            broadcast_index(out_midx, out_shape, b_shape, b_midx)    | 
                                                                     | 
            # Zip function                                           | 
            a_idx = index_to_position(a_midx, a_strides)             | 
            b_idx = index_to_position(b_midx, b_strides)             | 
            out_pos = index_to_position(out_midx, out_strides)       | 
            out[out_pos] = fn(a_storage[a_idx], b_storage[b_idx])    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #6, #3, #4, #5).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--6 is a parallel loop
   +--3 --> rewritten as a serial loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (parallel)
   +--4 (parallel)
   +--5 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--6 (parallel)
   +--3 (serial)
   +--4 (serial)
   +--5 (serial)


 
Parallel region 0 (loop #6) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#6).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (226) is 
hoisted out of the parallel loop labelled #6 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: out_midx = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (227) is 
hoisted out of the parallel loop labelled #6 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: a_midx = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (228) is 
hoisted out of the parallel loop labelled #6 (it will be performed before the 
loop is executed and reused inside the loop):
   Allocation:: b_midx = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (265)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (265) 
--------------------------------------------------------------------|loop #ID
    def _reduce(                                                    | 
        out: Storage,                                               | 
        out_shape: Shape,                                           | 
        out_strides: Strides,                                       | 
        a_storage: Storage,                                         | 
        a_shape: Shape,                                             | 
        a_strides: Strides,                                         | 
        reduce_dim: int,                                            | 
    ) -> None:                                                      | 
        """Optimized reduction function."""                         | 
        # Total number of elements in the output tensor             | 
        out_size = int(np.prod(out_shape))--------------------------| #7
                                                                    | 
        # Parallel loop over the output tensor                      | 
        for ordinal in prange(out_size):----------------------------| #8
            # Local index buffers                                   | 
            out_index = np.zeros_like(out_shape, dtype=np.int32)    | 
            a_index = np.zeros_like(a_shape, dtype=np.int32)        | 
                                                                    | 
            # Convert ordinal to multi-dimensional index            | 
            to_index(ordinal, out_shape, out_index)                 | 
                                                                    | 
            # Compute the base position in the input tensor         | 
            result = 0.0                                            | 
            to_index(ordinal, out_shape, a_index)                   | 
            a_index[reduce_dim] = 0                                 | 
            base_pos = index_to_position(a_index, a_strides)        | 
                                                                    | 
            # Reduce along the specified dimension                  | 
            result = a_storage[base_pos]                            | 
            for r in range(1, a_shape[reduce_dim]):                 | 
                a_index[reduce_dim] = r                             | 
                pos = index_to_position(a_index, a_strides)         | 
                result = fn(result, a_storage[pos])                 | 
                                                                    | 
            # Write the reduced value to the output tensor          | 
            out_pos = index_to_position(out_index, out_strides)     | 
            out[out_pos] = result                                   | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #7, #8).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (307)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/dev/Documents/Courses/MLE/mod3-devashar13/minitorch/fast_ops.py (307) 
----------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                            | 
    out: Storage,                                                                       | 
    out_shape: Shape,                                                                   | 
    out_strides: Strides,                                                               | 
    a_storage: Storage,                                                                 | 
    a_shape: Shape,                                                                     | 
    a_strides: Strides,                                                                 | 
    b_storage: Storage,                                                                 | 
    b_shape: Shape,                                                                     | 
    b_strides: Strides,                                                                 | 
) -> None:                                                                              | 
    """NUMBA tensor matrix multiply function.                                           | 
                                                                                        | 
    Should work for any tensor shapes that broadcast as long as                         | 
                                                                                        | 
    ```                                                                                 | 
    assert a_shape[-1] == b_shape[-2]                                                   | 
    ```                                                                                 | 
                                                                                        | 
    Optimizations:                                                                      | 
                                                                                        | 
    * Outer loop in parallel                                                            | 
    * No index buffers or function calls                                                | 
    * Inner loop should have no global writes, 1 multiply.                              | 
                                                                                        | 
                                                                                        | 
    Args:                                                                               | 
    ----                                                                                | 
        out (Storage): storage for `out` tensor                                         | 
        out_shape (Shape): shape for `out` tensor                                       | 
        out_strides (Strides): strides for `out` tensor                                 | 
        a_storage (Storage): storage for `a` tensor                                     | 
        a_shape (Shape): shape for `a` tensor                                           | 
        a_strides (Strides): strides for `a` tensor                                     | 
        b_storage (Storage): storage for `b` tensor                                     | 
        b_shape (Shape): shape for `b` tensor                                           | 
        b_strides (Strides): strides for `b` tensor                                     | 
                                                                                        | 
    Returns:                                                                            | 
    -------                                                                             | 
        None : Fills in `out`                                                           | 
                                                                                        | 
    """                                                                                 | 
    batch_size = out_shape[0]                                                           | 
    m, n = out_shape[1], out_shape[2]                                                   | 
    k = a_shape[-1]                                                                     | 
                                                                                        | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                              | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                              | 
                                                                                        | 
    for batch in prange(batch_size):----------------------------------------------------| #9
        for i in range(m):                                                              | 
            for j in range(n):                                                          | 
                result = 0.0                                                            | 
                for p in range(k):                                                      | 
                    a_index = (                                                         | 
                        batch * a_batch_stride + i * a_strides[1] + p * a_strides[2]    | 
                    )                                                                   | 
                    b_index = (                                                         | 
                        batch * b_batch_stride + p * b_strides[1] + j * b_strides[2]    | 
                    )                                                                   | 
                    result += a_storage[a_index] * b_storage[b_index]                   | 
                out_index = (                                                           | 
                    batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]    | 
                )                                                                       | 
                out[out_index] = result                                                 | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #9).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```
## Simple

### CPU
```
xEpoch  0  loss  6.859353801651763 correct 37
Epoch  10  loss  2.119963433342432 correct 47
Epoch  20  loss  1.9529842114961804 correct 47
Epoch  30  loss  1.0066644791771815 correct 50
Epoch  40  loss  0.683804589152195 correct 50
Epoch  50  loss  1.0440357688924415 correct 50
Epoch  60  loss  0.25614868845384425 correct 50
Epoch  70  loss  0.6173386338929369 correct 50
Epoch  80  loss  0.5250106875077628 correct 50
Epoch  90  loss  1.0481256180077945 correct 50
Epoch  100  loss  0.2699705650982966 correct 50
Epoch  110  loss  0.08676540762196434 correct 50
Epoch  120  loss  0.32237135404072703 correct 49
Epoch  130  loss  1.4246895636854315 correct 49
Epoch  140  loss  1.1272429322434196 correct 50
Epoch  150  loss  0.21604711983170283 correct 49
Epoch  160  loss  0.24692690976345946 correct 50
Epoch  170  loss  0.6738101208615117 correct 50
Epoch  180  loss  0.9082739523774165 correct 49
Epoch  190  loss  0.12343355534558016 correct 50
Epoch  200  loss  0.021076288498978437 correct 50
Epoch  210  loss  0.011490949154679719 correct 50
Epoch  220  loss  0.40763178136468825 correct 50
Epoch  230  loss  0.020074717608689244 correct 50
Epoch  240  loss  0.552405745074888 correct 50
Epoch  250  loss  0.5232574703216215 correct 50
Epoch  260  loss  0.5158832865928588 correct 50
Epoch  270  loss  0.35846856243495834 correct 50
Epoch  280  loss  0.15700129091420503 correct 50
Epoch  290  loss  0.21306420352240366 correct 50
Epoch  300  loss  0.008525390310832534 correct 50
Epoch  310  loss  0.26100215779067987 correct 50
Epoch  320  loss  0.00714107449466756 correct 50
Epoch  330  loss  0.31603320554128217 correct 50
Epoch  340  loss  0.021351099345657387 correct 50
Epoch  350  loss  0.002949007230761728 correct 50
Epoch  360  loss  0.46917668456025186 correct 50
Epoch  370  loss  0.3415226314879312 correct 50
Epoch  380  loss  0.19643740118436287 correct 50
Epoch  390  loss  0.09907294637535476 correct 50
Epoch  400  loss  0.6858414467606524 correct 50
Epoch  410  loss  0.14395805834881029 correct 50
Epoch  420  loss  0.09677767623243362 correct 50
Epoch  430  loss  0.10398892405488114 correct 50
Epoch  440  loss  0.09592007117530718 correct 50
Epoch  450  loss  0.500525940093333 correct 50
Epoch  460  loss  0.08831294597455214 correct 50
Epoch  470  loss  0.12616031865751703 correct 50
Epoch  480  loss  0.013545728521224832 correct 50
Epoch  490  loss  0.06000339283593174 correct 50
Average epoch time: 0.11s
```

### GPU

```
Epoch  0  loss  4.890260219573975 correct 35
Epoch  10  loss  3.8056774139404297 correct 40
Epoch  20  loss  4.426930904388428 correct 39
Epoch  30  loss  3.682494640350342 correct 45
Epoch  40  loss  3.98262357711792 correct 47
Epoch  50  loss  1.7545974254608154 correct 43
Epoch  60  loss  1.9271996021270752 correct 49
Epoch  70  loss  2.5528080463409424 correct 50
Epoch  80  loss  2.142118453979492 correct 50
Epoch  90  loss  1.5315567255020142 correct 50
Epoch  100  loss  0.8329392075538635 correct 50
Epoch  110  loss  0.7505874037742615 correct 50
Epoch  120  loss  1.2793817520141602 correct 50
Epoch  130  loss  1.4951648712158203 correct 50
Epoch  140  loss  1.0486363172531128 correct 50
Epoch  150  loss  0.693936288356781 correct 49
Epoch  160  loss  0.6659274101257324 correct 50
Epoch  170  loss  0.9229369163513184 correct 50
Epoch  180  loss  0.37151655554771423 correct 50
Epoch  190  loss  0.6887441277503967 correct 50
Epoch  200  loss  0.8120929002761841 correct 50
Epoch  210  loss  0.4308871626853943 correct 50
Epoch  220  loss  0.4619863033294678 correct 50
Epoch  230  loss  0.5590780377388 correct 50
Epoch  240  loss  0.3896164298057556 correct 50
Epoch  250  loss  0.10480395704507828 correct 50
Epoch  260  loss  0.33956992626190186 correct 50
Epoch  270  loss  0.2548990845680237 correct 50
Epoch  280  loss  0.6117724776268005 correct 50
Epoch  290  loss  0.4453679025173187 correct 50
Epoch  300  loss  0.1713268756866455 correct 50
Epoch  310  loss  0.6529228091239929 correct 50
Epoch  320  loss  0.5536174774169922 correct 50
Epoch  330  loss  0.1773928701877594 correct 50
Epoch  340  loss  0.5438095927238464 correct 50
Epoch  350  loss  0.20595793426036835 correct 50
Epoch  360  loss  0.2884258031845093 correct 50
Epoch  370  loss  0.5301516652107239 correct 50
Epoch  380  loss  0.44830435514450073 correct 50
Epoch  390  loss  0.49049630761146545 correct 50
Epoch  400  loss  0.30065858364105225 correct 50
Epoch  410  loss  0.39431288838386536 correct 50
Epoch  420  loss  0.17657622694969177 correct 50
Epoch  430  loss  0.39264222979545593 correct 50
Epoch  440  loss  0.06274643540382385 correct 50
Epoch  450  loss  0.18131962418556213 correct 50
Epoch  460  loss  0.715281069278717 correct 50
Epoch  470  loss  0.1721380054950714 correct 50
Epoch  480  loss  0.10999253392219543 correct 50
Epoch  490  loss  0.058572690933942795 correct 50
Average epoch time: 1.1408s
```
## SPLIT
### CPU

```
Epoch  0  loss  6.362771314754857 correct 32
Epoch  10  loss  5.6790133075282405 correct 41
Epoch  20  loss  2.9210571654362725 correct 47
Epoch  30  loss  2.8316699528737006 correct 46
Epoch  40  loss  1.6308267489459025 correct 47
Epoch  50  loss  0.9701177124827631 correct 47
Epoch  60  loss  1.4108897403641398 correct 47
Epoch  70  loss  3.865989888780713 correct 46
Epoch  80  loss  1.0584614701749722 correct 49
Epoch  90  loss  0.8524943063945097 correct 49
Epoch  100  loss  0.6210901641758515 correct 50
Epoch  110  loss  1.6147158312491172 correct 48
Epoch  120  loss  1.24314918082502 correct 50
Epoch  130  loss  0.4780738731115387 correct 49
Epoch  140  loss  1.2701345463773437 correct 49
Epoch  150  loss  0.6099687475430069 correct 49
Epoch  160  loss  1.1468693011836206 correct 49
Epoch  170  loss  1.079717031743134 correct 49
Epoch  180  loss  0.226342247539728 correct 50
Epoch  190  loss  0.9468749866796392 correct 48
Epoch  200  loss  0.4701905029590427 correct 49
Epoch  210  loss  0.24389144440707086 correct 50
Epoch  220  loss  0.7787401468992137 correct 50
Epoch  230  loss  0.9577880970778685 correct 50
Epoch  240  loss  1.0027314516837105 correct 50
Epoch  250  loss  0.39797214876404585 correct 50
Epoch  260  loss  0.06664538436016018 correct 50
Epoch  270  loss  0.10200633972976476 correct 50
Epoch  280  loss  1.0945633810957762 correct 50
Epoch  290  loss  0.5980207823860302 correct 50
Epoch  300  loss  0.047263986814343156 correct 50
Epoch  310  loss  0.6534623221894993 correct 50
Epoch  320  loss  0.2548684034155102 correct 50
Epoch  330  loss  0.38519049028076163 correct 50
Epoch  340  loss  0.01074340837915888 correct 50
Epoch  350  loss  0.1687189599066227 correct 50
Epoch  360  loss  0.6092652815731897 correct 50
Epoch  370  loss  0.027069098493746724 correct 50
Epoch  380  loss  0.8479263237160003 correct 50
Epoch  390  loss  0.6510434874908579 correct 50
Epoch  400  loss  0.1575293856291939 correct 50
Epoch  410  loss  0.3788109033623939 correct 50
Epoch  420  loss  0.14378979101317482 correct 50
Epoch  430  loss  0.5092677735820782 correct 50
Epoch  440  loss  0.7076665816445489 correct 50
Epoch  450  loss  0.37762730916470394 correct 50
Epoch  460  loss  0.176897819497023 correct 50
Epoch  470  loss  0.17622652715179138 correct 50
Epoch  480  loss  0.4052598447778219 correct 50
Epoch  490  loss  0.014657538208778412 correct 50

Avg Time Taken: 0.09348
```


### GPU


```
Epoch  0  loss  4.890260219573975 correct 35
Epoch  10  loss  3.8056774139404297 correct 40
Epoch  20  loss  4.426930904388428 correct 39
Epoch  30  loss  3.682494640350342 correct 45
Epoch  40  loss  3.98262357711792 correct 47
Epoch  50  loss  1.7545974254608154 correct 43
Epoch  60  loss  1.9271996021270752 correct 49
Epoch  70  loss  2.5528080463409424 correct 50
Epoch  80  loss  2.142118453979492 correct 50
Epoch  90  loss  1.5315567255020142 correct 50
Epoch  100  loss  0.8329392075538635 correct 50
Epoch  110  loss  0.7505874037742615 correct 50
Epoch  120  loss  1.2793817520141602 correct 50
Epoch  130  loss  1.4951648712158203 correct 50
Epoch  140  loss  1.0486363172531128 correct 50
Epoch  150  loss  0.693936288356781 correct 49
Epoch  160  loss  0.6659274101257324 correct 50
Epoch  170  loss  0.9229369163513184 correct 50
Epoch  180  loss  0.37151655554771423 correct 50
Epoch  190  loss  0.6887441277503967 correct 50
Epoch  200  loss  0.8120929002761841 correct 50
Epoch  210  loss  0.4308871626853943 correct 50
Epoch  220  loss  0.4619863033294678 correct 50
Epoch  230  loss  0.5590780377388 correct 50
Epoch  240  loss  0.3896164298057556 correct 50
Epoch  250  loss  0.10480395704507828 correct 50
Epoch  260  loss  0.33956992626190186 correct 50
Epoch  270  loss  0.2548990845680237 correct 50
Epoch  280  loss  0.6117724776268005 correct 50
Epoch  290  loss  0.4453679025173187 correct 50
Epoch  300  loss  0.1713268756866455 correct 50
Epoch  310  loss  0.6529228091239929 correct 50
Epoch  320  loss  0.5536174774169922 correct 50
Epoch  330  loss  0.1773928701877594 correct 50
Epoch  340  loss  0.5438095927238464 correct 50
Epoch  350  loss  0.20595793426036835 correct 50
Epoch  360  loss  0.2884258031845093 correct 50
Epoch  370  loss  0.5301516652107239 correct 50
Epoch  380  loss  0.44830435514450073 correct 50
Epoch  390  loss  0.49049630761146545 correct 50
Epoch  400  loss  0.30065858364105225 correct 50
Epoch  410  loss  0.39431288838386536 correct 50
Epoch  420  loss  0.17657622694969177 correct 50
Epoch  430  loss  0.39264222979545593 correct 50
Epoch  440  loss  0.06274643540382385 correct 50
Epoch  450  loss  0.18131962418556213 correct 50
Epoch  460  loss  0.715281069278717 correct 50
Epoch  470  loss  0.1721380054950714 correct 50
Epoch  480  loss  0.10999253392219543 correct 50
Epoch  490  loss  0.058572690933942795 correct 50
Avg Time: 1.494
```


## XOR

### CPU

```
Epoch  0  loss  7.759738822043825 correct 24
Epoch  10  loss  4.386852086470951 correct 38
Epoch  20  loss  3.3563058095804816 correct 38
Epoch  30  loss  6.02376216472036 correct 39
Epoch  40  loss  2.6654334912869633 correct 40
Epoch  50  loss  3.25232932982148 correct 41
Epoch  60  loss  1.8946660655529532 correct 46
Epoch  70  loss  2.811825212262136 correct 49
Epoch  80  loss  2.2200442246733374 correct 49
Epoch  90  loss  2.580557720240743 correct 50
Epoch  100  loss  2.423146719395849 correct 49
Epoch  110  loss  1.9860622841148283 correct 50
Epoch  120  loss  0.8948029957245945 correct 50
Epoch  130  loss  0.7235615078740039 correct 50
Epoch  140  loss  1.1901693615310107 correct 48
Epoch  150  loss  1.7910682123153414 correct 50
Epoch  160  loss  0.4550720705049517 correct 50
Epoch  170  loss  2.2893495049651236 correct 50
Epoch  180  loss  1.6727308543162103 correct 50
Epoch  190  loss  0.9761825557151076 correct 49
Epoch  200  loss  1.010241940237773 correct 50
Epoch  210  loss  0.5470424022314366 correct 50
Epoch  220  loss  0.5630777056936705 correct 49
Epoch  230  loss  1.2905333947671283 correct 50
Epoch  240  loss  1.2521133414455998 correct 50
Epoch  250  loss  0.5080270419154282 correct 49
Epoch  260  loss  1.5266205780879525 correct 50
Epoch  270  loss  0.8187808287424965 correct 50
Epoch  280  loss  0.7468296309501107 correct 50
Epoch  290  loss  1.3340873666982713 correct 50
Epoch  300  loss  0.33393129937748484 correct 50
Epoch  310  loss  0.03603164193664099 correct 50
Epoch  320  loss  0.8309179844918262 correct 50
Epoch  330  loss  1.5033507669305326 correct 50
Epoch  340  loss  0.27475235792062846 correct 50
Epoch  350  loss  0.42176175605345007 correct 50
Epoch  360  loss  0.2390351813807896 correct 50
Epoch  370  loss  0.3924467976164481 correct 50
Epoch  380  loss  0.4787261708202372 correct 50
Epoch  390  loss  0.10900351277633527 correct 50
Epoch  400  loss  0.20675158877983443 correct 50
Epoch  410  loss  0.4984136725407552 correct 50
Epoch  420  loss  0.3922198181307187 correct 50
Epoch  430  loss  0.14676077613055724 correct 50
Epoch  440  loss  0.3851528301379606 correct 50
Epoch  450  loss  1.173451814468468 correct 50
Epoch  460  loss  0.7582227827001108 correct 50
Epoch  470  loss  0.31439486979517095 correct 50
Epoch  480  loss  0.14464350894961905 correct 50
Epoch  490  loss  0.151089918824766 correct 50
Average epoch time: 0.1455s
```


### GPU

```
Epoch  0  loss  7.008540630340576 correct 27
Epoch  10  loss  5.647858619689941 correct 46
Epoch  20  loss  3.5495707988739014 correct 39
Epoch  30  loss  2.6364169120788574 correct 47
Epoch  40  loss  2.3781070709228516 correct 46
Epoch  50  loss  2.4486498832702637 correct 46
Epoch  60  loss  2.1358144283294678 correct 46
Epoch  70  loss  0.651817262172699 correct 47
Epoch  80  loss  2.431823253631592 correct 47
Epoch  90  loss  0.5314754247665405 correct 48
Epoch  100  loss  1.6281704902648926 correct 49
Epoch  110  loss  0.6019164323806763 correct 48
Epoch  120  loss  1.5700054168701172 correct 49
Epoch  130  loss  0.4345915913581848 correct 48
Epoch  140  loss  0.8809261918067932 correct 47
Epoch  150  loss  1.3215770721435547 correct 49
Epoch  160  loss  0.9875820875167847 correct 49
Epoch  170  loss  1.2124288082122803 correct 49
Epoch  180  loss  1.0757232904434204 correct 49
Epoch  190  loss  1.1639775037765503 correct 49
Epoch  200  loss  0.8500983715057373 correct 49
Epoch  210  loss  1.1678591966629028 correct 49
Epoch  220  loss  0.3081539273262024 correct 49
Epoch  230  loss  0.37135034799575806 correct 49
Epoch  240  loss  0.5202480554580688 correct 49
Epoch  250  loss  1.4400681257247925 correct 49
Epoch  260  loss  0.7637478113174438 correct 49
Epoch  270  loss  1.4448573589324951 correct 47
Epoch  280  loss  0.6683148741722107 correct 49
Epoch  290  loss  0.6279646754264832 correct 49
Epoch  300  loss  0.5454403758049011 correct 49
Epoch  310  loss  0.6764714121818542 correct 48
Epoch  320  loss  1.890180230140686 correct 50
Epoch  330  loss  0.5891304016113281 correct 49
Epoch  340  loss  0.16515445709228516 correct 49
Epoch  350  loss  0.3830890953540802 correct 49
Epoch  360  loss  2.1644222736358643 correct 49
Epoch  370  loss  0.18221846222877502 correct 49
Epoch  380  loss  0.3609005808830261 correct 48
Epoch  390  loss  0.3873912990093231 correct 49
Epoch  400  loss  1.772550106048584 correct 49
Epoch  410  loss  0.8563373684883118 correct 49
Epoch  420  loss  1.9346450567245483 correct 49
Epoch  430  loss  0.013318486511707306 correct 49
Epoch  440  loss  0.4364946782588959 correct 49
Epoch  450  loss  0.5091127753257751 correct 49
Epoch  460  loss  0.06711467355489731 correct 49
Epoch  470  loss  1.603857398033142 correct 49
Epoch  480  loss  0.40449684858322144 correct 50
Epoch  490  loss  0.6303718090057373 correct 49
Average epoch time: 1.6029s
```


## Complex Model

### CPU

```
Epoch  0  loss  18.018813244715258 correct 45
Epoch  10  loss  1.4407767301481909 correct 48
Epoch  20  loss  0.10197095951991919 correct 47
Epoch  30  loss  0.03198936971441677 correct 48
Epoch  40  loss  0.07908132157477962 correct 48
Epoch  50  loss  1.1066960086719049 correct 48
Epoch  60  loss  0.00855393247488822 correct 48
Epoch  70  loss  3.236008695684685e-05 correct 50
Epoch  80  loss  0.030454534381947663 correct 48
Epoch  90  loss  1.5317802851139457 correct 48
Epoch  100  loss  8.106058580146704e-05 correct 48
Epoch  110  loss  0.10940974813654583 correct 49
Epoch  120  loss  0.03317980498769379 correct 47
Epoch  130  loss  0.0852078539148857 correct 48
Epoch  140  loss  0.14224246967010598 correct 50
Epoch  150  loss  8.981029612618874e-05 correct 48
Epoch  160  loss  2.813153030311847 correct 47
Epoch  170  loss  0.01833190642873016 correct 49
Epoch  180  loss  0.0006756923768855193 correct 47
Epoch  190  loss  2.34206313262967e-05 correct 49
Epoch  200  loss  0.4098148630263007 correct 50
Epoch  210  loss  0.0009134811846579723 correct 50
Epoch  220  loss  1.262596101063761 correct 49
Epoch  230  loss  0.0005511014492973443 correct 49
Epoch  240  loss  0.0004748989861495016 correct 50
Epoch  250  loss  1.7122804895865151 correct 50
Epoch  260  loss  0.7730297671158951 correct 50
Epoch  270  loss  1.7017698240116885 correct 48
Epoch  280  loss  7.67604430150697e-05 correct 48
Epoch  290  loss  0.00015720861462211567 correct 49
Epoch  300  loss  0.8056369870740894 correct 49
Epoch  310  loss  0.10279552482644214 correct 49
Epoch  320  loss  0.01556637997951324 correct 50
Epoch  330  loss  0.002527094185062111 correct 49
Epoch  340  loss  1.2591259405871478 correct 49
Epoch  350  loss  0.015929144578272967 correct 49
Epoch  360  loss  1.0037090312829283e-05 correct 50
Epoch  370  loss  0.012980926773377616 correct 48
Epoch  380  loss  1.4393521134656873 correct 49
Epoch  390  loss  0.0026635601466635005 correct 49
Epoch  400  loss  1.0113775349418013 correct 49
Epoch  410  loss  0.9404050353682976 correct 49
Epoch  420  loss  0.003565233999630641 correct 49
Epoch  430  loss  4.8786595690565394e-05 correct 50
Epoch  440  loss  0.10198509558967381 correct 49
Epoch  450  loss  0.4486131365762257 correct 50
Epoch  460  loss  1.695096763775839e-05 correct 49
Epoch  470  loss  1.4488161722668076 correct 49
Epoch  480  loss  0.00022395072171274726 correct 49
Epoch  490  loss  0.01064588992768443 correct 50
Average epoch time: 1.0527s
```


### GPU

```
Epoch  0  loss  12.434595203892872 correct 49
Epoch  10  loss  0.16716457040994884 correct 50
Epoch  20  loss  0.5097334128210291 correct 50
Epoch  30  loss  0.25608493487570855 correct 50
Epoch  40  loss  0.014890785819805966 correct 50
Epoch  50  loss  0.0005454371779703797 correct 50
Epoch  60  loss  0.02148426823776925 correct 50
Epoch  70  loss  0.010595965788121886 correct 50
Epoch  80  loss  0.138397081648725 correct 50
Epoch  90  loss  0.015941497067279542 correct 50
Epoch  100  loss  0.00030403173639929394 correct 50
Epoch  110  loss  0.03556956012216768 correct 50
Epoch  120  loss  0.166824434675735 correct 50
Epoch  130  loss  0.15866788919768968 correct 50
Epoch  140  loss  0.1302862535842229 correct 50
Epoch  150  loss  0.14090029586399008 correct 50
Epoch  160  loss  0.031372438544128745 correct 50
Epoch  170  loss  0.011181479610845386 correct 50
Epoch  180  loss  0.0836905150093794 correct 50
Epoch  190  loss  0.003368679372432943 correct 50
Epoch  200  loss  0.07236112470946808 correct 50
Epoch  210  loss  0.09845491581591632 correct 50
Epoch  220  loss  0.007696474295048132 correct 50
Epoch  230  loss  0.002625519973470659 correct 50
Epoch  240  loss  0.00013719052421357474 correct 50
Epoch  250  loss  0.08866804103800664 correct 50
Epoch  260  loss  0.011851065154907948 correct 50
Epoch  270  loss  0.02714302386409289 correct 50
Epoch  280  loss  0.0019449737131134854 correct 50
Epoch  290  loss  0.004878948073740699 correct 50
Epoch  300  loss  0.0029344720627059928 correct 50
Epoch  310  loss  0.00848951363435851 correct 50
Epoch  320  loss  0.005746418394172495 correct 50
Epoch  330  loss  0.017057183727142575 correct 50
Epoch  340  loss  0.06893966855949192 correct 50
Epoch  350  loss  0.025671661695330562 correct 50
Epoch  360  loss  0.006529200928493615 correct 50
Epoch  370  loss  0.003447806980522418 correct 50
Epoch  380  loss  0.004381496256324896 correct 50
Epoch  390  loss  0.00124052448231311 correct 50
Epoch  400  loss  0.038602954995186636 correct 50
Epoch  410  loss  0.07225662275149672 correct 50
Epoch  420  loss  3.0112540857708322e-06 correct 50
Epoch  430  loss  0.008976667018654697 correct 50
Epoch  440  loss  0.05975644985256773 correct 50
Epoch  450  loss  0.005819504067015088 correct 50
Epoch  460  loss  0.05005716148101174 correct 50
Epoch  470  loss  0.01649317696522713 correct 50
Epoch  480  loss  0.06512389297820827 correct 50
Epoch  490  loss  0.06089542743193416 correct 50
Average epoch time: 1.7684s
```


## Matmul Time Graph

```
Timing summary
Size: 64
    fast: 0.00342
    gpu: 0.00567
Size: 128
    fast: 0.01409
    gpu: 0.01262
Size: 256
    fast: 0.09175
    gpu: 0.04925
Size: 512
    fast: 0.96728
    gpu: 0.18807
Size: 1024
    fast: 8.91312
    gpu: 0.90138

```

![image](https://github.com/user-attachments/assets/a79b8581-1ea3-4740-98dd-ffaff5f8968d)
