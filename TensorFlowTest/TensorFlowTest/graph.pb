
@
inputPlaceholder*
shape:˙˙˙˙˙˙˙˙˙*
dtype0
@
targetPlaceholder*
shape:˙˙˙˙˙˙˙˙˙
*
dtype0

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"     *
_class
loc:@dense/kernel*
dtype0
y
+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *HY½*
_class
loc:@dense/kernel*
dtype0
y
+dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *HY=*
_class
loc:@dense/kernel*
dtype0
Ċ
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*

seed *
T0*
_class
loc:@dense/kernel*
dtype0*
seed2 
ĥ
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel
À
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel
´
'dense/kernel/Initializer/random_uniformAddV2+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel

dense/kernel
VariableV2*
shape:
*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container 
§
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
U
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel
m
dense/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0
z

dense/bias
VariableV2*
shape:*
shared_name *
_class
loc:@dense/bias*
dtype0*
	container 

dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(
O
dense/bias/readIdentity
dense/bias*
T0*
_class
loc:@dense/bias
_
dense/MatMulMatMulinputdense/kernel/read*
transpose_b( *
T0*
transpose_a( 
W
dense/BiasAddBiasAdddense/MatMuldense/bias/read*
T0*
data_formatNHWC
&
hiddenReludense/BiasAdd*
T0

/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"      *!
_class
loc:@dense_1/kernel*
dtype0
}
-dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *óµ½*!
_class
loc:@dense_1/kernel*
dtype0
}
-dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *óµ=*!
_class
loc:@dense_1/kernel*
dtype0
Ë
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape*

seed *
T0*!
_class
loc:@dense_1/kernel*
dtype0*
seed2 
?
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel
È
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_1/kernel
ĵ
)dense_1/kernel/Initializer/random_uniformAddV2-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_1/kernel

dense_1/kernel
VariableV2*
shape:
*
shared_name *!
_class
loc:@dense_1/kernel*
dtype0*
	container 
Ż
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(
[
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel
q
dense_1/bias/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_1/bias*
dtype0
~
dense_1/bias
VariableV2*
shape:*
shared_name *
_class
loc:@dense_1/bias*
dtype0*
	container 

dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(
U
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias
d
dense_1/MatMulMatMulhiddendense_1/kernel/read*
transpose_b( *
T0*
transpose_a( 
]
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC
)
hidden2Reludense_1/BiasAdd*
T0

/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"   
   *!
_class
loc:@dense_2/kernel*
dtype0
}
-dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *ĜÊ?*!
_class
loc:@dense_2/kernel*
dtype0
}
-dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *ĜÊ>*!
_class
loc:@dense_2/kernel*
dtype0
Ë
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape*

seed *
T0*!
_class
loc:@dense_2/kernel*
dtype0*
seed2 
?
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel
È
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*
T0*!
_class
loc:@dense_2/kernel
ĵ
)dense_2/kernel/Initializer/random_uniformAddV2-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*
T0*!
_class
loc:@dense_2/kernel

dense_2/kernel
VariableV2*
shape:	
*
shared_name *!
_class
loc:@dense_2/kernel*
dtype0*
	container 
Ż
dense_2/kernel/AssignAssigndense_2/kernel)dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(
[
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel
p
dense_2/bias/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_2/bias*
dtype0
}
dense_2/bias
VariableV2*
shape:
*
shared_name *
_class
loc:@dense_2/bias*
dtype0*
	container 

dense_2/bias/AssignAssigndense_2/biasdense_2/bias/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(
U
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias
e
dense_2/MatMulMatMulhidden2dense_2/kernel/read*
transpose_b( *
T0*
transpose_a( 
]
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC
,
outputIdentitydense_2/BiasAdd*
T0
P
&softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0
Q
'softmax_cross_entropy_with_logits/ShapeShapeoutput*
T0*
out_type0
R
(softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0
S
)softmax_cross_entropy_with_logits/Shape_1Shapeoutput*
T0*
out_type0
Q
'softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0

%softmax_cross_entropy_with_logits/SubSub(softmax_cross_entropy_with_logits/Rank_1'softmax_cross_entropy_with_logits/Sub/y*
T0
z
-softmax_cross_entropy_with_logits/Slice/beginPack%softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N
Z
,softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0
Î
'softmax_cross_entropy_with_logits/SliceSlice)softmax_cross_entropy_with_logits/Shape_1-softmax_cross_entropy_with_logits/Slice/begin,softmax_cross_entropy_with_logits/Slice/size*
T0*
Index0
h
1softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
W
-softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0
Ŭ
(softmax_cross_entropy_with_logits/concatConcatV21softmax_cross_entropy_with_logits/concat/values_0'softmax_cross_entropy_with_logits/Slice-softmax_cross_entropy_with_logits/concat/axis*

Tidx0*
T0*
N
}
)softmax_cross_entropy_with_logits/ReshapeReshapeoutput(softmax_cross_entropy_with_logits/concat*
T0*
Tshape0
R
(softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0
S
)softmax_cross_entropy_with_logits/Shape_2Shapetarget*
T0*
out_type0
S
)softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0

'softmax_cross_entropy_with_logits/Sub_1Sub(softmax_cross_entropy_with_logits/Rank_2)softmax_cross_entropy_with_logits/Sub_1/y*
T0
~
/softmax_cross_entropy_with_logits/Slice_1/beginPack'softmax_cross_entropy_with_logits/Sub_1*
T0*

axis *
N
\
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0
Ô
)softmax_cross_entropy_with_logits/Slice_1Slice)softmax_cross_entropy_with_logits/Shape_2/softmax_cross_entropy_with_logits/Slice_1/begin.softmax_cross_entropy_with_logits/Slice_1/size*
T0*
Index0
j
3softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0
Y
/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0
ċ
*softmax_cross_entropy_with_logits/concat_1ConcatV23softmax_cross_entropy_with_logits/concat_1/values_0)softmax_cross_entropy_with_logits/Slice_1/softmax_cross_entropy_with_logits/concat_1/axis*

Tidx0*
T0*
N

+softmax_cross_entropy_with_logits/Reshape_1Reshapetarget*softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0
£
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits)softmax_cross_entropy_with_logits/Reshape+softmax_cross_entropy_with_logits/Reshape_1*
T0
S
)softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0

'softmax_cross_entropy_with_logits/Sub_2Sub&softmax_cross_entropy_with_logits/Rank)softmax_cross_entropy_with_logits/Sub_2/y*
T0
]
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0
}
.softmax_cross_entropy_with_logits/Slice_2/sizePack'softmax_cross_entropy_with_logits/Sub_2*
T0*

axis *
N
Ò
)softmax_cross_entropy_with_logits/Slice_2Slice'softmax_cross_entropy_with_logits/Shape/softmax_cross_entropy_with_logits/Slice_2/begin.softmax_cross_entropy_with_logits/Slice_2/size*
T0*
Index0

+softmax_cross_entropy_with_logits/Reshape_2Reshape!softmax_cross_entropy_with_logits)softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0
^
gradients/ShapeShape+softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0
F
gradients/grad_ys_0/ConstConst*
valueB
 *  ?*
dtype0
b
gradients/grad_ys_0Fillgradients/Shapegradients/grad_ys_0/Const*
T0*

index_type0

@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape!softmax_cross_entropy_with_logits*
T0*
out_type0
ğ
Bgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapegradients/grad_ys_0@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*
Tshape0
O
gradients/zeros_like	ZerosLike#softmax_cross_entropy_with_logits:1*
T0
r
?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
?
;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*

Tdim0*
T0
Ĥ
4gradients/softmax_cross_entropy_with_logits_grad/mulMul;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims#softmax_cross_entropy_with_logits:1*
T0
}
;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax)softmax_cross_entropy_with_logits/Reshape*
T0

4gradients/softmax_cross_entropy_with_logits_grad/NegNeg;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0
t
Agradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0
ç
=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*

Tdim0*
T0
ğ
6gradients/softmax_cross_entropy_with_logits_grad/mul_1Mul=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_14gradients/softmax_cross_entropy_with_logits_grad/Neg*
T0
ı
Agradients/softmax_cross_entropy_with_logits_grad/tuple/group_depsNoOp5^gradients/softmax_cross_entropy_with_logits_grad/mul7^gradients/softmax_cross_entropy_with_logits_grad/mul_1
Ħ
Igradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependencyIdentity4gradients/softmax_cross_entropy_with_logits_grad/mulB^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/softmax_cross_entropy_with_logits_grad/mul
§
Kgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1Identity6gradients/softmax_cross_entropy_with_logits_grad/mul_1B^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_with_logits_grad/mul_1
h
>gradients/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapeoutput*
T0*
out_type0
í
@gradients/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapeIgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency>gradients/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0

*gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
T0*
data_formatNHWC
§
/gradients/dense_2/BiasAdd_grad/tuple/group_depsNoOp+^gradients/dense_2/BiasAdd_grad/BiasAddGradA^gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape

7gradients/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape
ë
9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_2/BiasAdd_grad/BiasAddGrad0^gradients/dense_2/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/dense_2/BiasAdd_grad/BiasAddGrad
Ğ
$gradients/dense_2/MatMul_grad/MatMulMatMul7gradients/dense_2/BiasAdd_grad/tuple/control_dependencydense_2/kernel/read*
transpose_b(*
T0*
transpose_a( 
Ħ
&gradients/dense_2/MatMul_grad/MatMul_1MatMulhidden27gradients/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(

.gradients/dense_2/MatMul_grad/tuple/group_depsNoOp%^gradients/dense_2/MatMul_grad/MatMul'^gradients/dense_2/MatMul_grad/MatMul_1
Û
6gradients/dense_2/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_2/MatMul_grad/MatMul/^gradients/dense_2/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense_2/MatMul_grad/MatMul
á
8gradients/dense_2/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_2/MatMul_grad/MatMul_1/^gradients/dense_2/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/dense_2/MatMul_grad/MatMul_1
u
gradients/hidden2_grad/ReluGradReluGrad6gradients/dense_2/MatMul_grad/tuple/control_dependencyhidden2*
T0
z
*gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGradgradients/hidden2_grad/ReluGrad*
T0*
data_formatNHWC

/gradients/dense_1/BiasAdd_grad/tuple/group_depsNoOp+^gradients/dense_1/BiasAdd_grad/BiasAddGrad ^gradients/hidden2_grad/ReluGrad
Ó
7gradients/dense_1/BiasAdd_grad/tuple/control_dependencyIdentitygradients/hidden2_grad/ReluGrad0^gradients/dense_1/BiasAdd_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/hidden2_grad/ReluGrad
ë
9gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity*gradients/dense_1/BiasAdd_grad/BiasAddGrad0^gradients/dense_1/BiasAdd_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/dense_1/BiasAdd_grad/BiasAddGrad
Ğ
$gradients/dense_1/MatMul_grad/MatMulMatMul7gradients/dense_1/BiasAdd_grad/tuple/control_dependencydense_1/kernel/read*
transpose_b(*
T0*
transpose_a( 
 
&gradients/dense_1/MatMul_grad/MatMul_1MatMulhidden7gradients/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(

.gradients/dense_1/MatMul_grad/tuple/group_depsNoOp%^gradients/dense_1/MatMul_grad/MatMul'^gradients/dense_1/MatMul_grad/MatMul_1
Û
6gradients/dense_1/MatMul_grad/tuple/control_dependencyIdentity$gradients/dense_1/MatMul_grad/MatMul/^gradients/dense_1/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense_1/MatMul_grad/MatMul
á
8gradients/dense_1/MatMul_grad/tuple/control_dependency_1Identity&gradients/dense_1/MatMul_grad/MatMul_1/^gradients/dense_1/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/dense_1/MatMul_grad/MatMul_1
s
gradients/hidden_grad/ReluGradReluGrad6gradients/dense_1/MatMul_grad/tuple/control_dependencyhidden*
T0
w
(gradients/dense/BiasAdd_grad/BiasAddGradBiasAddGradgradients/hidden_grad/ReluGrad*
T0*
data_formatNHWC

-gradients/dense/BiasAdd_grad/tuple/group_depsNoOp)^gradients/dense/BiasAdd_grad/BiasAddGrad^gradients/hidden_grad/ReluGrad
Í
5gradients/dense/BiasAdd_grad/tuple/control_dependencyIdentitygradients/hidden_grad/ReluGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/hidden_grad/ReluGrad
?
7gradients/dense/BiasAdd_grad/tuple/control_dependency_1Identity(gradients/dense/BiasAdd_grad/BiasAddGrad.^gradients/dense/BiasAdd_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/dense/BiasAdd_grad/BiasAddGrad
?
"gradients/dense/MatMul_grad/MatMulMatMul5gradients/dense/BiasAdd_grad/tuple/control_dependencydense/kernel/read*
transpose_b(*
T0*
transpose_a( 

$gradients/dense/MatMul_grad/MatMul_1MatMulinput5gradients/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(

,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
Ó
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul
Ù
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1
e
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@dense/bias*
dtype0
v
beta1_power
VariableV2*
shape: *
shared_name *
_class
loc:@dense/bias*
dtype0*
	container 

beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(
Q
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@dense/bias
e
beta2_power/initial_valueConst*
valueB
 *w??*
_class
loc:@dense/bias*
dtype0
v
beta2_power
VariableV2*
shape: *
shared_name *
_class
loc:@dense/bias*
dtype0*
	container 

beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(
Q
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@dense/bias

3dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"     *
_class
loc:@dense/kernel*
dtype0
w
)dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@dense/kernel*
dtype0
Ç
#dense/kernel/Adam/Initializer/zerosFill3dense/kernel/Adam/Initializer/zeros/shape_as_tensor)dense/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@dense/kernel

dense/kernel/Adam
VariableV2*
shape:
*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container 
­
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
_
dense/kernel/Adam/readIdentitydense/kernel/Adam*
T0*
_class
loc:@dense/kernel

5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"     *
_class
loc:@dense/kernel*
dtype0
y
+dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *
_class
loc:@dense/kernel*
dtype0
Í
%dense/kernel/Adam_1/Initializer/zerosFill5dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor+dense/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*
_class
loc:@dense/kernel

dense/kernel/Adam_1
VariableV2*
shape:
*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container 
³
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
c
dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1*
T0*
_class
loc:@dense/kernel
r
!dense/bias/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0

dense/bias/Adam
VariableV2*
shape:*
shared_name *
_class
loc:@dense/bias*
dtype0*
	container 
?
dense/bias/Adam/AssignAssigndense/bias/Adam!dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(
Y
dense/bias/Adam/readIdentitydense/bias/Adam*
T0*
_class
loc:@dense/bias
t
#dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@dense/bias*
dtype0

dense/bias/Adam_1
VariableV2*
shape:*
shared_name *
_class
loc:@dense/bias*
dtype0*
	container 
Ğ
dense/bias/Adam_1/AssignAssigndense/bias/Adam_1#dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(
]
dense/bias/Adam_1/readIdentitydense/bias/Adam_1*
T0*
_class
loc:@dense/bias

5dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@dense_1/kernel*
dtype0
{
+dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@dense_1/kernel*
dtype0
Ï
%dense_1/kernel/Adam/Initializer/zerosFill5dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor+dense_1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@dense_1/kernel

dense_1/kernel/Adam
VariableV2*
shape:
*
shared_name *!
_class
loc:@dense_1/kernel*
dtype0*
	container 
µ
dense_1/kernel/Adam/AssignAssigndense_1/kernel/Adam%dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(
e
dense_1/kernel/Adam/readIdentitydense_1/kernel/Adam*
T0*!
_class
loc:@dense_1/kernel

7dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *!
_class
loc:@dense_1/kernel*
dtype0
}
-dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@dense_1/kernel*
dtype0
Ġ
'dense_1/kernel/Adam_1/Initializer/zerosFill7dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor-dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@dense_1/kernel

dense_1/kernel/Adam_1
VariableV2*
shape:
*
shared_name *!
_class
loc:@dense_1/kernel*
dtype0*
	container 
ğ
dense_1/kernel/Adam_1/AssignAssigndense_1/kernel/Adam_1'dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(
i
dense_1/kernel/Adam_1/readIdentitydense_1/kernel/Adam_1*
T0*!
_class
loc:@dense_1/kernel
v
#dense_1/bias/Adam/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_1/bias*
dtype0

dense_1/bias/Adam
VariableV2*
shape:*
shared_name *
_class
loc:@dense_1/bias*
dtype0*
	container 
­
dense_1/bias/Adam/AssignAssigndense_1/bias/Adam#dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(
_
dense_1/bias/Adam/readIdentitydense_1/bias/Adam*
T0*
_class
loc:@dense_1/bias
x
%dense_1/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_class
loc:@dense_1/bias*
dtype0

dense_1/bias/Adam_1
VariableV2*
shape:*
shared_name *
_class
loc:@dense_1/bias*
dtype0*
	container 
³
dense_1/bias/Adam_1/AssignAssigndense_1/bias/Adam_1%dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(
c
dense_1/bias/Adam_1/readIdentitydense_1/bias/Adam_1*
T0*
_class
loc:@dense_1/bias

5dense_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"   
   *!
_class
loc:@dense_2/kernel*
dtype0
{
+dense_2/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@dense_2/kernel*
dtype0
Ï
%dense_2/kernel/Adam/Initializer/zerosFill5dense_2/kernel/Adam/Initializer/zeros/shape_as_tensor+dense_2/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@dense_2/kernel

dense_2/kernel/Adam
VariableV2*
shape:	
*
shared_name *!
_class
loc:@dense_2/kernel*
dtype0*
	container 
µ
dense_2/kernel/Adam/AssignAssigndense_2/kernel/Adam%dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(
e
dense_2/kernel/Adam/readIdentitydense_2/kernel/Adam*
T0*!
_class
loc:@dense_2/kernel

7dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"   
   *!
_class
loc:@dense_2/kernel*
dtype0
}
-dense_2/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *!
_class
loc:@dense_2/kernel*
dtype0
Ġ
'dense_2/kernel/Adam_1/Initializer/zerosFill7dense_2/kernel/Adam_1/Initializer/zeros/shape_as_tensor-dense_2/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*!
_class
loc:@dense_2/kernel

dense_2/kernel/Adam_1
VariableV2*
shape:	
*
shared_name *!
_class
loc:@dense_2/kernel*
dtype0*
	container 
ğ
dense_2/kernel/Adam_1/AssignAssigndense_2/kernel/Adam_1'dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(
i
dense_2/kernel/Adam_1/readIdentitydense_2/kernel/Adam_1*
T0*!
_class
loc:@dense_2/kernel
u
#dense_2/bias/Adam/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_2/bias*
dtype0

dense_2/bias/Adam
VariableV2*
shape:
*
shared_name *
_class
loc:@dense_2/bias*
dtype0*
	container 
­
dense_2/bias/Adam/AssignAssigndense_2/bias/Adam#dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(
_
dense_2/bias/Adam/readIdentitydense_2/bias/Adam*
T0*
_class
loc:@dense_2/bias
w
%dense_2/bias/Adam_1/Initializer/zerosConst*
valueB
*    *
_class
loc:@dense_2/bias*
dtype0

dense_2/bias/Adam_1
VariableV2*
shape:
*
shared_name *
_class
loc:@dense_2/bias*
dtype0*
	container 
³
dense_2/bias/Adam_1/AssignAssigndense_2/bias/Adam_1%dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(
c
dense_2/bias/Adam_1/readIdentitydense_2/bias/Adam_1*
T0*
_class
loc:@dense_2/bias
@
train/learning_rateConst*
valueB
 *o:*
dtype0
8
train/beta1Const*
valueB
 *fff?*
dtype0
8
train/beta2Const*
valueB
 *w??*
dtype0
:
train/epsilonConst*
valueB
 *wÌ+2*
dtype0
Ñ
#train/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon6gradients/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
use_nesterov( 
È
!train/update_dense/bias/ApplyAdam	ApplyAdam
dense/biasdense/bias/Adamdense/bias/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon7gradients/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/bias*
use_nesterov( 
Ŭ
%train/update_dense_1/kernel/ApplyAdam	ApplyAdamdense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon8gradients/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_1/kernel*
use_nesterov( 
Ô
#train/update_dense_1/bias/ApplyAdam	ApplyAdamdense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon9gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_1/bias*
use_nesterov( 
Ŭ
%train/update_dense_2/kernel/ApplyAdam	ApplyAdamdense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon8gradients/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*!
_class
loc:@dense_2/kernel*
use_nesterov( 
Ô
#train/update_dense_2/bias/ApplyAdam	ApplyAdamdense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1beta1_power/readbeta2_power/readtrain/learning_ratetrain/beta1train/beta2train/epsilon9gradients/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense_2/bias*
use_nesterov( 
½
	train/mulMulbeta1_power/readtrain/beta1"^train/update_dense/bias/ApplyAdam$^train/update_dense/kernel/ApplyAdam$^train/update_dense_1/bias/ApplyAdam&^train/update_dense_1/kernel/ApplyAdam$^train/update_dense_2/bias/ApplyAdam&^train/update_dense_2/kernel/ApplyAdam*
T0*
_class
loc:@dense/bias

train/AssignAssignbeta1_power	train/mul*
use_locking( *
T0*
_class
loc:@dense/bias*
validate_shape(
ż
train/mul_1Mulbeta2_power/readtrain/beta2"^train/update_dense/bias/ApplyAdam$^train/update_dense/kernel/ApplyAdam$^train/update_dense_1/bias/ApplyAdam&^train/update_dense_1/kernel/ApplyAdam$^train/update_dense_2/bias/ApplyAdam&^train/update_dense_2/kernel/ApplyAdam*
T0*
_class
loc:@dense/bias

train/Assign_1Assignbeta2_powertrain/mul_1*
use_locking( *
T0*
_class
loc:@dense/bias*
validate_shape(

trainNoOp^train/Assign^train/Assign_1"^train/update_dense/bias/ApplyAdam$^train/update_dense/kernel/ApplyAdam$^train/update_dense_1/bias/ApplyAdam&^train/update_dense_1/kernel/ApplyAdam$^train/update_dense_2/bias/ApplyAdam&^train/update_dense_2/kernel/ApplyAdam

initNoOp^beta1_power/Assign^beta2_power/Assign^dense/bias/Adam/Assign^dense/bias/Adam_1/Assign^dense/bias/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign^dense_1/bias/Adam/Assign^dense_1/bias/Adam_1/Assign^dense_1/bias/Assign^dense_1/kernel/Adam/Assign^dense_1/kernel/Adam_1/Assign^dense_1/kernel/Assign^dense_2/bias/Adam/Assign^dense_2/bias/Adam_1/Assign^dense_2/bias/Assign^dense_2/kernel/Adam/Assign^dense_2/kernel/Adam_1/Assign^dense_2/kernel/Assign
A
save/filename/inputConst*
valueB Bmodel*
dtype0
V
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0
M

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0
Ş
save/SaveV2/tensor_namesConst*ù
valueïBìBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1Bdense_2/biasBdense_2/bias/AdamBdense_2/bias/Adam_1Bdense_2/kernelBdense_2/kernel/AdamBdense_2/kernel/Adam_1*
dtype0
o
save/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0
á
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_power
dense/biasdense/bias/Adamdense/bias/Adam_1dense/kerneldense/kernel/Adamdense/kernel/Adam_1dense_1/biasdense_1/bias/Adamdense_1/bias/Adam_1dense_1/kerneldense_1/kernel/Adamdense_1/kernel/Adam_1dense_2/biasdense_2/bias/Adamdense_2/bias/Adam_1dense_2/kerneldense_2/kernel/Adamdense_2/kernel/Adam_1*"
dtypes
2
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
ĵ
save/RestoreV2/tensor_namesConst"/device:CPU:0*ù
valueïBìBbeta1_powerBbeta2_powerB
dense/biasBdense/bias/AdamBdense/bias/Adam_1Bdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bdense_1/biasBdense_1/bias/AdamBdense_1/bias/Adam_1Bdense_1/kernelBdense_1/kernel/AdamBdense_1/kernel/Adam_1Bdense_2/biasBdense_2/bias/AdamBdense_2/bias/Adam_1Bdense_2/kernelBdense_2/kernel/AdamBdense_2/kernel/Adam_1*
dtype0

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*"
dtypes
2

save/AssignAssignbeta1_powersave/RestoreV2*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(

save/Assign_1Assignbeta2_powersave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(

save/Assign_2Assign
dense/biassave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(

save/Assign_3Assigndense/bias/Adamsave/RestoreV2:3*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(

save/Assign_4Assigndense/bias/Adam_1save/RestoreV2:4*
use_locking(*
T0*
_class
loc:@dense/bias*
validate_shape(

save/Assign_5Assigndense/kernelsave/RestoreV2:5*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(

save/Assign_6Assigndense/kernel/Adamsave/RestoreV2:6*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(

save/Assign_7Assigndense/kernel/Adam_1save/RestoreV2:7*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(

save/Assign_8Assigndense_1/biassave/RestoreV2:8*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(

save/Assign_9Assigndense_1/bias/Adamsave/RestoreV2:9*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(

save/Assign_10Assigndense_1/bias/Adam_1save/RestoreV2:10*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(

save/Assign_11Assigndense_1/kernelsave/RestoreV2:11*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(

save/Assign_12Assigndense_1/kernel/Adamsave/RestoreV2:12*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(

save/Assign_13Assigndense_1/kernel/Adam_1save/RestoreV2:13*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(

save/Assign_14Assigndense_2/biassave/RestoreV2:14*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(

save/Assign_15Assigndense_2/bias/Adamsave/RestoreV2:15*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(

save/Assign_16Assigndense_2/bias/Adam_1save/RestoreV2:16*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(

save/Assign_17Assigndense_2/kernelsave/RestoreV2:17*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(

save/Assign_18Assigndense_2/kernel/Adamsave/RestoreV2:18*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(

save/Assign_19Assigndense_2/kernel/Adam_1save/RestoreV2:19*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(
à
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9"ż