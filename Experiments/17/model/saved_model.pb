¿%
É£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-0-gb36436b8Å
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

sequential/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namesequential/conv2d/kernel

,sequential/conv2d/kernel/Read/ReadVariableOpReadVariableOpsequential/conv2d/kernel*&
_output_shapes
: *
dtype0

sequential/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namesequential/conv2d/bias
}
*sequential/conv2d/bias/Read/ReadVariableOpReadVariableOpsequential/conv2d/bias*
_output_shapes
: *
dtype0
 
$sequential/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$sequential/batch_normalization/gamma

8sequential/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp$sequential/batch_normalization/gamma*
_output_shapes
: *
dtype0

#sequential/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#sequential/batch_normalization/beta

7sequential/batch_normalization/beta/Read/ReadVariableOpReadVariableOp#sequential/batch_normalization/beta*
_output_shapes
: *
dtype0

sequential_1/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *-
shared_namesequential_1/conv2d_1/kernel

0sequential_1/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_1/kernel*&
_output_shapes
:  *
dtype0

sequential_1/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namesequential_1/conv2d_1/bias

.sequential_1/conv2d_1/bias/Read/ReadVariableOpReadVariableOpsequential_1/conv2d_1/bias*
_output_shapes
: *
dtype0
¨
(sequential_1/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(sequential_1/batch_normalization_1/gamma
¡
<sequential_1/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp(sequential_1/batch_normalization_1/gamma*
_output_shapes
: *
dtype0
¦
'sequential_1/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'sequential_1/batch_normalization_1/beta

;sequential_1/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp'sequential_1/batch_normalization_1/beta*
_output_shapes
: *
dtype0

sequential_3/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*-
shared_namesequential_3/conv2d_3/kernel

0sequential_3/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_3/kernel*&
_output_shapes
:@@*
dtype0

sequential_3/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namesequential_3/conv2d_3/bias

.sequential_3/conv2d_3/bias/Read/ReadVariableOpReadVariableOpsequential_3/conv2d_3/bias*
_output_shapes
:@*
dtype0
¨
(sequential_3/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(sequential_3/batch_normalization_3/gamma
¡
<sequential_3/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp(sequential_3/batch_normalization_3/gamma*
_output_shapes
:@*
dtype0
¦
'sequential_3/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'sequential_3/batch_normalization_3/beta

;sequential_3/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp'sequential_3/batch_normalization_3/beta*
_output_shapes
:@*
dtype0

sequential_5/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namesequential_5/conv2d_5/kernel

0sequential_5/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpsequential_5/conv2d_5/kernel*(
_output_shapes
:*
dtype0

sequential_5/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namesequential_5/conv2d_5/bias

.sequential_5/conv2d_5/bias/Read/ReadVariableOpReadVariableOpsequential_5/conv2d_5/bias*
_output_shapes	
:*
dtype0
©
(sequential_5/batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(sequential_5/batch_normalization_5/gamma
¢
<sequential_5/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOp(sequential_5/batch_normalization_5/gamma*
_output_shapes	
:*
dtype0
§
'sequential_5/batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'sequential_5/batch_normalization_5/beta
 
;sequential_5/batch_normalization_5/beta/Read/ReadVariableOpReadVariableOp'sequential_5/batch_normalization_5/beta*
_output_shapes	
:*
dtype0

sequential_6/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namesequential_6/dense/kernel

-sequential_6/dense/kernel/Read/ReadVariableOpReadVariableOpsequential_6/dense/kernel* 
_output_shapes
:
*
dtype0

sequential_6/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namesequential_6/dense/bias

+sequential_6/dense/bias/Read/ReadVariableOpReadVariableOpsequential_6/dense/bias*
_output_shapes
:*
dtype0
¬
*sequential/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*sequential/batch_normalization/moving_mean
¥
>sequential/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp*sequential/batch_normalization/moving_mean*
_output_shapes
: *
dtype0
´
.sequential/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.sequential/batch_normalization/moving_variance
­
Bsequential/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp.sequential/batch_normalization/moving_variance*
_output_shapes
: *
dtype0
´
.sequential_1/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.sequential_1/batch_normalization_1/moving_mean
­
Bsequential_1/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp.sequential_1/batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
¼
2sequential_1/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42sequential_1/batch_normalization_1/moving_variance
µ
Fsequential_1/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp2sequential_1/batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
´
.sequential_3/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.sequential_3/batch_normalization_3/moving_mean
­
Bsequential_3/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp.sequential_3/batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
¼
2sequential_3/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42sequential_3/batch_normalization_3/moving_variance
µ
Fsequential_3/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp2sequential_3/batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
µ
.sequential_5/batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.sequential_5/batch_normalization_5/moving_mean
®
Bsequential_5/batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp.sequential_5/batch_normalization_5/moving_mean*
_output_shapes	
:*
dtype0
½
2sequential_5/batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42sequential_5/batch_normalization_5/moving_variance
¶
Fsequential_5/batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp2sequential_5/batch_normalization_5/moving_variance*
_output_shapes	
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
¢
Adam/sequential/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/sequential/conv2d/kernel/m

3Adam/sequential/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/kernel/m*&
_output_shapes
: *
dtype0

Adam/sequential/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/sequential/conv2d/bias/m

1Adam/sequential/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/bias/m*
_output_shapes
: *
dtype0
®
+Adam/sequential/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/sequential/batch_normalization/gamma/m
§
?Adam/sequential/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp+Adam/sequential/batch_normalization/gamma/m*
_output_shapes
: *
dtype0
¬
*Adam/sequential/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/sequential/batch_normalization/beta/m
¥
>Adam/sequential/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOp*Adam/sequential/batch_normalization/beta/m*
_output_shapes
: *
dtype0
ª
#Adam/sequential_1/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *4
shared_name%#Adam/sequential_1/conv2d_1/kernel/m
£
7Adam/sequential_1/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/conv2d_1/kernel/m*&
_output_shapes
:  *
dtype0

!Adam/sequential_1/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/sequential_1/conv2d_1/bias/m

5Adam/sequential_1/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/conv2d_1/bias/m*
_output_shapes
: *
dtype0
¶
/Adam/sequential_1/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/Adam/sequential_1/batch_normalization_1/gamma/m
¯
CAdam/sequential_1/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp/Adam/sequential_1/batch_normalization_1/gamma/m*
_output_shapes
: *
dtype0
´
.Adam/sequential_1/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.Adam/sequential_1/batch_normalization_1/beta/m
­
BAdam/sequential_1/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp.Adam/sequential_1/batch_normalization_1/beta/m*
_output_shapes
: *
dtype0
ª
#Adam/sequential_3/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#Adam/sequential_3/conv2d_3/kernel/m
£
7Adam/sequential_3/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_3/conv2d_3/kernel/m*&
_output_shapes
:@@*
dtype0

!Adam/sequential_3/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/sequential_3/conv2d_3/bias/m

5Adam/sequential_3/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_3/conv2d_3/bias/m*
_output_shapes
:@*
dtype0
¶
/Adam/sequential_3/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/Adam/sequential_3/batch_normalization_3/gamma/m
¯
CAdam/sequential_3/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp/Adam/sequential_3/batch_normalization_3/gamma/m*
_output_shapes
:@*
dtype0
´
.Adam/sequential_3/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/sequential_3/batch_normalization_3/beta/m
­
BAdam/sequential_3/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp.Adam/sequential_3/batch_normalization_3/beta/m*
_output_shapes
:@*
dtype0
¬
#Adam/sequential_5/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/sequential_5/conv2d_5/kernel/m
¥
7Adam/sequential_5/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_5/conv2d_5/kernel/m*(
_output_shapes
:*
dtype0

!Adam/sequential_5/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/sequential_5/conv2d_5/bias/m

5Adam/sequential_5/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_5/conv2d_5/bias/m*
_output_shapes	
:*
dtype0
·
/Adam/sequential_5/batch_normalization_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adam/sequential_5/batch_normalization_5/gamma/m
°
CAdam/sequential_5/batch_normalization_5/gamma/m/Read/ReadVariableOpReadVariableOp/Adam/sequential_5/batch_normalization_5/gamma/m*
_output_shapes	
:*
dtype0
µ
.Adam/sequential_5/batch_normalization_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/sequential_5/batch_normalization_5/beta/m
®
BAdam/sequential_5/batch_normalization_5/beta/m/Read/ReadVariableOpReadVariableOp.Adam/sequential_5/batch_normalization_5/beta/m*
_output_shapes	
:*
dtype0

 Adam/sequential_6/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/sequential_6/dense/kernel/m

4Adam/sequential_6/dense/kernel/m/Read/ReadVariableOpReadVariableOp Adam/sequential_6/dense/kernel/m* 
_output_shapes
:
*
dtype0

Adam/sequential_6/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/sequential_6/dense/bias/m

2Adam/sequential_6/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential_6/dense/bias/m*
_output_shapes
:*
dtype0
¢
Adam/sequential/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/sequential/conv2d/kernel/v

3Adam/sequential/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/kernel/v*&
_output_shapes
: *
dtype0

Adam/sequential/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nameAdam/sequential/conv2d/bias/v

1Adam/sequential/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/conv2d/bias/v*
_output_shapes
: *
dtype0
®
+Adam/sequential/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/sequential/batch_normalization/gamma/v
§
?Adam/sequential/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp+Adam/sequential/batch_normalization/gamma/v*
_output_shapes
: *
dtype0
¬
*Adam/sequential/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/sequential/batch_normalization/beta/v
¥
>Adam/sequential/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOp*Adam/sequential/batch_normalization/beta/v*
_output_shapes
: *
dtype0
ª
#Adam/sequential_1/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *4
shared_name%#Adam/sequential_1/conv2d_1/kernel/v
£
7Adam/sequential_1/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_1/conv2d_1/kernel/v*&
_output_shapes
:  *
dtype0

!Adam/sequential_1/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/sequential_1/conv2d_1/bias/v

5Adam/sequential_1/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_1/conv2d_1/bias/v*
_output_shapes
: *
dtype0
¶
/Adam/sequential_1/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *@
shared_name1/Adam/sequential_1/batch_normalization_1/gamma/v
¯
CAdam/sequential_1/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp/Adam/sequential_1/batch_normalization_1/gamma/v*
_output_shapes
: *
dtype0
´
.Adam/sequential_1/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *?
shared_name0.Adam/sequential_1/batch_normalization_1/beta/v
­
BAdam/sequential_1/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp.Adam/sequential_1/batch_normalization_1/beta/v*
_output_shapes
: *
dtype0
ª
#Adam/sequential_3/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#Adam/sequential_3/conv2d_3/kernel/v
£
7Adam/sequential_3/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_3/conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0

!Adam/sequential_3/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/sequential_3/conv2d_3/bias/v

5Adam/sequential_3/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_3/conv2d_3/bias/v*
_output_shapes
:@*
dtype0
¶
/Adam/sequential_3/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/Adam/sequential_3/batch_normalization_3/gamma/v
¯
CAdam/sequential_3/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp/Adam/sequential_3/batch_normalization_3/gamma/v*
_output_shapes
:@*
dtype0
´
.Adam/sequential_3/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/sequential_3/batch_normalization_3/beta/v
­
BAdam/sequential_3/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp.Adam/sequential_3/batch_normalization_3/beta/v*
_output_shapes
:@*
dtype0
¬
#Adam/sequential_5/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/sequential_5/conv2d_5/kernel/v
¥
7Adam/sequential_5/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_5/conv2d_5/kernel/v*(
_output_shapes
:*
dtype0

!Adam/sequential_5/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/sequential_5/conv2d_5/bias/v

5Adam/sequential_5/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_5/conv2d_5/bias/v*
_output_shapes	
:*
dtype0
·
/Adam/sequential_5/batch_normalization_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adam/sequential_5/batch_normalization_5/gamma/v
°
CAdam/sequential_5/batch_normalization_5/gamma/v/Read/ReadVariableOpReadVariableOp/Adam/sequential_5/batch_normalization_5/gamma/v*
_output_shapes	
:*
dtype0
µ
.Adam/sequential_5/batch_normalization_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/sequential_5/batch_normalization_5/beta/v
®
BAdam/sequential_5/batch_normalization_5/beta/v/Read/ReadVariableOpReadVariableOp.Adam/sequential_5/batch_normalization_5/beta/v*
_output_shapes	
:*
dtype0

 Adam/sequential_6/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" Adam/sequential_6/dense/kernel/v

4Adam/sequential_6/dense/kernel/v/Read/ReadVariableOpReadVariableOp Adam/sequential_6/dense/kernel/v* 
_output_shapes
:
*
dtype0

Adam/sequential_6/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/sequential_6/dense/bias/v

2Adam/sequential_6/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential_6/dense/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
È
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value÷Bó Bë

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
­
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
º
layer_with_weights-0
layer-0
layer-1
 layer_with_weights-1
 layer-2
!layer-3
"trainable_variables
#	variables
$regularization_losses
%	keras_api
R
&trainable_variables
'	variables
(regularization_losses
)	keras_api
R
*trainable_variables
+	variables
,regularization_losses
-	keras_api
º
.layer_with_weights-0
.layer-0
/layer-1
0layer_with_weights-1
0layer-2
1layer-3
2trainable_variables
3	variables
4regularization_losses
5	keras_api
R
6trainable_variables
7	variables
8regularization_losses
9	keras_api
R
:trainable_variables
;	variables
<regularization_losses
=	keras_api
º
>layer_with_weights-0
>layer-0
?layer-1
@layer_with_weights-1
@layer-2
Alayer-3
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
R
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api

Jlayer-0
Klayer_with_weights-0
Klayer-1
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
¨
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_rateUmäVmåWmæXmçYmèZmé[mê\më]mì^mí_mî`mïamðbmñcmòdmóemôfmõUvöVv÷WvøXvùYvúZvû[vü\vý]vþ^vÿ_v`vavbvcvdvevfv

U0
V1
W2
X3
Y4
Z5
[6
\7
]8
^9
_10
`11
a12
b13
c14
d15
e16
f17
Æ
U0
V1
W2
X3
g4
h5
Y6
Z7
[8
\9
i10
j11
]12
^13
_14
`15
k16
l17
a18
b19
c20
d21
m22
n23
e24
f25
 
­
ometrics
trainable_variables

players
qlayer_metrics
rnon_trainable_variables
	variables
regularization_losses
slayer_regularization_losses
 
|
t_inbound_nodes

Ukernel
Vbias
utrainable_variables
v	variables
wregularization_losses
x	keras_api
f
y_inbound_nodes
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
¯
~_inbound_nodes
axis
	Wgamma
Xbeta
gmoving_mean
hmoving_variance
trainable_variables
	variables
regularization_losses
	keras_api

U0
V1
W2
X3
*
U0
V1
W2
X3
g4
h5
 
²
metrics
trainable_variables
layers
layer_metrics
non_trainable_variables
	variables
regularization_losses
 layer_regularization_losses
 
 
 
²
metrics
trainable_variables
layers
layer_metrics
non_trainable_variables
	variables
regularization_losses
 layer_regularization_losses

_inbound_nodes

Ykernel
Zbias
trainable_variables
	variables
regularization_losses
	keras_api
k
_inbound_nodes
trainable_variables
	variables
regularization_losses
	keras_api
±
_inbound_nodes
	axis
	[gamma
\beta
imoving_mean
jmoving_variance
trainable_variables
	variables
regularization_losses
	keras_api
k
_inbound_nodes
trainable_variables
 	variables
¡regularization_losses
¢	keras_api

Y0
Z1
[2
\3
*
Y0
Z1
[2
\3
i4
j5
 
²
£metrics
"trainable_variables
¤layers
¥layer_metrics
¦non_trainable_variables
#	variables
$regularization_losses
 §layer_regularization_losses
 
 
 
²
¨metrics
&trainable_variables
©layers
ªlayer_metrics
«non_trainable_variables
'	variables
(regularization_losses
 ¬layer_regularization_losses
 
 
 
²
­metrics
*trainable_variables
®layers
¯layer_metrics
°non_trainable_variables
+	variables
,regularization_losses
 ±layer_regularization_losses

²_inbound_nodes

]kernel
^bias
³trainable_variables
´	variables
µregularization_losses
¶	keras_api
k
·_inbound_nodes
¸trainable_variables
¹	variables
ºregularization_losses
»	keras_api
±
¼_inbound_nodes
	½axis
	_gamma
`beta
kmoving_mean
lmoving_variance
¾trainable_variables
¿	variables
Àregularization_losses
Á	keras_api
k
Â_inbound_nodes
Ãtrainable_variables
Ä	variables
Åregularization_losses
Æ	keras_api

]0
^1
_2
`3
*
]0
^1
_2
`3
k4
l5
 
²
Çmetrics
2trainable_variables
Èlayers
Élayer_metrics
Ênon_trainable_variables
3	variables
4regularization_losses
 Ëlayer_regularization_losses
 
 
 
²
Ìmetrics
6trainable_variables
Ílayers
Îlayer_metrics
Ïnon_trainable_variables
7	variables
8regularization_losses
 Ðlayer_regularization_losses
 
 
 
²
Ñmetrics
:trainable_variables
Òlayers
Ólayer_metrics
Ônon_trainable_variables
;	variables
<regularization_losses
 Õlayer_regularization_losses

Ö_inbound_nodes

akernel
bbias
×trainable_variables
Ø	variables
Ùregularization_losses
Ú	keras_api
k
Û_inbound_nodes
Ütrainable_variables
Ý	variables
Þregularization_losses
ß	keras_api
±
à_inbound_nodes
	áaxis
	cgamma
dbeta
mmoving_mean
nmoving_variance
âtrainable_variables
ã	variables
äregularization_losses
å	keras_api
k
æ_inbound_nodes
çtrainable_variables
è	variables
éregularization_losses
ê	keras_api

a0
b1
c2
d3
*
a0
b1
c2
d3
m4
n5
 
²
ëmetrics
Btrainable_variables
ìlayers
ílayer_metrics
înon_trainable_variables
C	variables
Dregularization_losses
 ïlayer_regularization_losses
 
 
 
²
ðmetrics
Ftrainable_variables
ñlayers
òlayer_metrics
ónon_trainable_variables
G	variables
Hregularization_losses
 ôlayer_regularization_losses
k
õ_inbound_nodes
ötrainable_variables
÷	variables
øregularization_losses
ù	keras_api

ú_inbound_nodes

ekernel
fbias
ûtrainable_variables
ü	variables
ýregularization_losses
þ	keras_api

e0
f1

e0
f1
 
²
ÿmetrics
Ltrainable_variables
layers
layer_metrics
non_trainable_variables
M	variables
Nregularization_losses
 layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEsequential/conv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEsequential/conv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE$sequential/batch_normalization/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#sequential/batch_normalization/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEsequential_1/conv2d_1/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEsequential_1/conv2d_1/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE(sequential_1/batch_normalization_1/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'sequential_1/batch_normalization_1/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEsequential_3/conv2d_3/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEsequential_3/conv2d_3/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE(sequential_3/batch_normalization_3/gamma1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'sequential_3/batch_normalization_3/beta1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEsequential_5/conv2d_5/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEsequential_5/conv2d_5/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE(sequential_5/batch_normalization_5/gamma1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE'sequential_5/batch_normalization_5/beta1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEsequential_6/dense/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEsequential_6/dense/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE*sequential/batch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE.sequential/batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.sequential_1/batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2sequential_1/batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.sequential_3/batch_normalization_3/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2sequential_3/batch_normalization_3/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.sequential_5/batch_normalization_5/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2sequential_5/batch_normalization_5/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE

0
1
V
0
1
2
3
4
5
6
7
	8

9
10
11
 
8
g0
h1
i2
j3
k4
l5
m6
n7
 
 

U0
V1

U0
V1
 
²
metrics
utrainable_variables
layers
layer_metrics
non_trainable_variables
v	variables
wregularization_losses
 layer_regularization_losses
 
 
 
 
²
metrics
ztrainable_variables
layers
layer_metrics
non_trainable_variables
{	variables
|regularization_losses
 layer_regularization_losses
 
 

W0
X1

W0
X1
g2
h3
 
µ
metrics
trainable_variables
layers
layer_metrics
non_trainable_variables
	variables
regularization_losses
 layer_regularization_losses
 

0
1
2
 

g0
h1
 
 
 
 
 
 
 

Y0
Z1

Y0
Z1
 
µ
metrics
trainable_variables
layers
layer_metrics
non_trainable_variables
	variables
regularization_losses
 layer_regularization_losses
 
 
 
 
µ
metrics
trainable_variables
layers
layer_metrics
non_trainable_variables
	variables
regularization_losses
 layer_regularization_losses
 
 

[0
\1

[0
\1
i2
j3
 
µ
metrics
trainable_variables
 layers
¡layer_metrics
¢non_trainable_variables
	variables
regularization_losses
 £layer_regularization_losses
 
 
 
 
µ
¤metrics
trainable_variables
¥layers
¦layer_metrics
§non_trainable_variables
 	variables
¡regularization_losses
 ¨layer_regularization_losses
 

0
1
 2
!3
 

i0
j1
 
 
 
 
 
 
 
 
 
 
 
 

]0
^1

]0
^1
 
µ
©metrics
³trainable_variables
ªlayers
«layer_metrics
¬non_trainable_variables
´	variables
µregularization_losses
 ­layer_regularization_losses
 
 
 
 
µ
®metrics
¸trainable_variables
¯layers
°layer_metrics
±non_trainable_variables
¹	variables
ºregularization_losses
 ²layer_regularization_losses
 
 

_0
`1

_0
`1
k2
l3
 
µ
³metrics
¾trainable_variables
´layers
µlayer_metrics
¶non_trainable_variables
¿	variables
Àregularization_losses
 ·layer_regularization_losses
 
 
 
 
µ
¸metrics
Ãtrainable_variables
¹layers
ºlayer_metrics
»non_trainable_variables
Ä	variables
Åregularization_losses
 ¼layer_regularization_losses
 

.0
/1
02
13
 

k0
l1
 
 
 
 
 
 
 
 
 
 
 
 

a0
b1

a0
b1
 
µ
½metrics
×trainable_variables
¾layers
¿layer_metrics
Ànon_trainable_variables
Ø	variables
Ùregularization_losses
 Álayer_regularization_losses
 
 
 
 
µ
Âmetrics
Ütrainable_variables
Ãlayers
Älayer_metrics
Ånon_trainable_variables
Ý	variables
Þregularization_losses
 Ælayer_regularization_losses
 
 

c0
d1

c0
d1
m2
n3
 
µ
Çmetrics
âtrainable_variables
Èlayers
Élayer_metrics
Ênon_trainable_variables
ã	variables
äregularization_losses
 Ëlayer_regularization_losses
 
 
 
 
µ
Ìmetrics
çtrainable_variables
Ílayers
Îlayer_metrics
Ïnon_trainable_variables
è	variables
éregularization_losses
 Ðlayer_regularization_losses
 

>0
?1
@2
A3
 

m0
n1
 
 
 
 
 
 
 
 
 
 
µ
Ñmetrics
ötrainable_variables
Òlayers
Ólayer_metrics
Ônon_trainable_variables
÷	variables
øregularization_losses
 Õlayer_regularization_losses
 

e0
f1

e0
f1
 
µ
Ömetrics
ûtrainable_variables
×layers
Ølayer_metrics
Ùnon_trainable_variables
ü	variables
ýregularization_losses
 Úlayer_regularization_losses
 

J0
K1
 
 
 
8

Ûtotal

Ücount
Ý	variables
Þ	keras_api
I

ßtotal

àcount
á
_fn_kwargs
â	variables
ã	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 

g0
h1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

i0
j1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

k0
l1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

m0
n1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Û0
Ü1

Ý	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ß0
à1

â	variables

VARIABLE_VALUEAdam/sequential/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/sequential/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+Adam/sequential/batch_normalization/gamma/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/sequential/batch_normalization/beta/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/sequential_1/conv2d_1/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/sequential_1/conv2d_1/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_1/batch_normalization_1/gamma/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_1/batch_normalization_1/beta/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/sequential_3/conv2d_3/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/sequential_3/conv2d_3/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_3/batch_normalization_3/gamma/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_3/batch_normalization_3/beta/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/sequential_5/conv2d_5/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/sequential_5/conv2d_5/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_5/batch_normalization_5/gamma/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_5/batch_normalization_5/beta/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/sequential_6/dense/kernel/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/sequential_6/dense/bias/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/sequential/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/sequential/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+Adam/sequential/batch_normalization/gamma/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/sequential/batch_normalization/beta/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/sequential_1/conv2d_1/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/sequential_1/conv2d_1/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_1/batch_normalization_1/gamma/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_1/batch_normalization_1/beta/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/sequential_3/conv2d_3/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/sequential_3/conv2d_3/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_3/batch_normalization_3/gamma/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_3/batch_normalization_3/beta/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE#Adam/sequential_5/conv2d_5/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/sequential_5/conv2d_5/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_5/batch_normalization_5/gamma/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_5/batch_normalization_5/beta/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/sequential_6/dense/kernel/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/sequential_6/dense/bias/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
È

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1sequential/conv2d/kernelsequential/conv2d/bias$sequential/batch_normalization/gamma#sequential/batch_normalization/beta*sequential/batch_normalization/moving_mean.sequential/batch_normalization/moving_variancesequential_1/conv2d_1/kernelsequential_1/conv2d_1/bias(sequential_1/batch_normalization_1/gamma'sequential_1/batch_normalization_1/beta.sequential_1/batch_normalization_1/moving_mean2sequential_1/batch_normalization_1/moving_variancesequential_3/conv2d_3/kernelsequential_3/conv2d_3/bias(sequential_3/batch_normalization_3/gamma'sequential_3/batch_normalization_3/beta.sequential_3/batch_normalization_3/moving_mean2sequential_3/batch_normalization_3/moving_variancesequential_5/conv2d_5/kernelsequential_5/conv2d_5/bias(sequential_5/batch_normalization_5/gamma'sequential_5/batch_normalization_5/beta.sequential_5/batch_normalization_5/moving_mean2sequential_5/batch_normalization_5/moving_variancesequential_6/dense/kernelsequential_6/dense/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_26234
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
á!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp,sequential/conv2d/kernel/Read/ReadVariableOp*sequential/conv2d/bias/Read/ReadVariableOp8sequential/batch_normalization/gamma/Read/ReadVariableOp7sequential/batch_normalization/beta/Read/ReadVariableOp0sequential_1/conv2d_1/kernel/Read/ReadVariableOp.sequential_1/conv2d_1/bias/Read/ReadVariableOp<sequential_1/batch_normalization_1/gamma/Read/ReadVariableOp;sequential_1/batch_normalization_1/beta/Read/ReadVariableOp0sequential_3/conv2d_3/kernel/Read/ReadVariableOp.sequential_3/conv2d_3/bias/Read/ReadVariableOp<sequential_3/batch_normalization_3/gamma/Read/ReadVariableOp;sequential_3/batch_normalization_3/beta/Read/ReadVariableOp0sequential_5/conv2d_5/kernel/Read/ReadVariableOp.sequential_5/conv2d_5/bias/Read/ReadVariableOp<sequential_5/batch_normalization_5/gamma/Read/ReadVariableOp;sequential_5/batch_normalization_5/beta/Read/ReadVariableOp-sequential_6/dense/kernel/Read/ReadVariableOp+sequential_6/dense/bias/Read/ReadVariableOp>sequential/batch_normalization/moving_mean/Read/ReadVariableOpBsequential/batch_normalization/moving_variance/Read/ReadVariableOpBsequential_1/batch_normalization_1/moving_mean/Read/ReadVariableOpFsequential_1/batch_normalization_1/moving_variance/Read/ReadVariableOpBsequential_3/batch_normalization_3/moving_mean/Read/ReadVariableOpFsequential_3/batch_normalization_3/moving_variance/Read/ReadVariableOpBsequential_5/batch_normalization_5/moving_mean/Read/ReadVariableOpFsequential_5/batch_normalization_5/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp3Adam/sequential/conv2d/kernel/m/Read/ReadVariableOp1Adam/sequential/conv2d/bias/m/Read/ReadVariableOp?Adam/sequential/batch_normalization/gamma/m/Read/ReadVariableOp>Adam/sequential/batch_normalization/beta/m/Read/ReadVariableOp7Adam/sequential_1/conv2d_1/kernel/m/Read/ReadVariableOp5Adam/sequential_1/conv2d_1/bias/m/Read/ReadVariableOpCAdam/sequential_1/batch_normalization_1/gamma/m/Read/ReadVariableOpBAdam/sequential_1/batch_normalization_1/beta/m/Read/ReadVariableOp7Adam/sequential_3/conv2d_3/kernel/m/Read/ReadVariableOp5Adam/sequential_3/conv2d_3/bias/m/Read/ReadVariableOpCAdam/sequential_3/batch_normalization_3/gamma/m/Read/ReadVariableOpBAdam/sequential_3/batch_normalization_3/beta/m/Read/ReadVariableOp7Adam/sequential_5/conv2d_5/kernel/m/Read/ReadVariableOp5Adam/sequential_5/conv2d_5/bias/m/Read/ReadVariableOpCAdam/sequential_5/batch_normalization_5/gamma/m/Read/ReadVariableOpBAdam/sequential_5/batch_normalization_5/beta/m/Read/ReadVariableOp4Adam/sequential_6/dense/kernel/m/Read/ReadVariableOp2Adam/sequential_6/dense/bias/m/Read/ReadVariableOp3Adam/sequential/conv2d/kernel/v/Read/ReadVariableOp1Adam/sequential/conv2d/bias/v/Read/ReadVariableOp?Adam/sequential/batch_normalization/gamma/v/Read/ReadVariableOp>Adam/sequential/batch_normalization/beta/v/Read/ReadVariableOp7Adam/sequential_1/conv2d_1/kernel/v/Read/ReadVariableOp5Adam/sequential_1/conv2d_1/bias/v/Read/ReadVariableOpCAdam/sequential_1/batch_normalization_1/gamma/v/Read/ReadVariableOpBAdam/sequential_1/batch_normalization_1/beta/v/Read/ReadVariableOp7Adam/sequential_3/conv2d_3/kernel/v/Read/ReadVariableOp5Adam/sequential_3/conv2d_3/bias/v/Read/ReadVariableOpCAdam/sequential_3/batch_normalization_3/gamma/v/Read/ReadVariableOpBAdam/sequential_3/batch_normalization_3/beta/v/Read/ReadVariableOp7Adam/sequential_5/conv2d_5/kernel/v/Read/ReadVariableOp5Adam/sequential_5/conv2d_5/bias/v/Read/ReadVariableOpCAdam/sequential_5/batch_normalization_5/gamma/v/Read/ReadVariableOpBAdam/sequential_5/batch_normalization_5/beta/v/Read/ReadVariableOp4Adam/sequential_6/dense/kernel/v/Read/ReadVariableOp2Adam/sequential_6/dense/bias/v/Read/ReadVariableOpConst*T
TinM
K2I	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__traced_save_28162
Ð
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratesequential/conv2d/kernelsequential/conv2d/bias$sequential/batch_normalization/gamma#sequential/batch_normalization/betasequential_1/conv2d_1/kernelsequential_1/conv2d_1/bias(sequential_1/batch_normalization_1/gamma'sequential_1/batch_normalization_1/betasequential_3/conv2d_3/kernelsequential_3/conv2d_3/bias(sequential_3/batch_normalization_3/gamma'sequential_3/batch_normalization_3/betasequential_5/conv2d_5/kernelsequential_5/conv2d_5/bias(sequential_5/batch_normalization_5/gamma'sequential_5/batch_normalization_5/betasequential_6/dense/kernelsequential_6/dense/bias*sequential/batch_normalization/moving_mean.sequential/batch_normalization/moving_variance.sequential_1/batch_normalization_1/moving_mean2sequential_1/batch_normalization_1/moving_variance.sequential_3/batch_normalization_3/moving_mean2sequential_3/batch_normalization_3/moving_variance.sequential_5/batch_normalization_5/moving_mean2sequential_5/batch_normalization_5/moving_variancetotalcounttotal_1count_1Adam/sequential/conv2d/kernel/mAdam/sequential/conv2d/bias/m+Adam/sequential/batch_normalization/gamma/m*Adam/sequential/batch_normalization/beta/m#Adam/sequential_1/conv2d_1/kernel/m!Adam/sequential_1/conv2d_1/bias/m/Adam/sequential_1/batch_normalization_1/gamma/m.Adam/sequential_1/batch_normalization_1/beta/m#Adam/sequential_3/conv2d_3/kernel/m!Adam/sequential_3/conv2d_3/bias/m/Adam/sequential_3/batch_normalization_3/gamma/m.Adam/sequential_3/batch_normalization_3/beta/m#Adam/sequential_5/conv2d_5/kernel/m!Adam/sequential_5/conv2d_5/bias/m/Adam/sequential_5/batch_normalization_5/gamma/m.Adam/sequential_5/batch_normalization_5/beta/m Adam/sequential_6/dense/kernel/mAdam/sequential_6/dense/bias/mAdam/sequential/conv2d/kernel/vAdam/sequential/conv2d/bias/v+Adam/sequential/batch_normalization/gamma/v*Adam/sequential/batch_normalization/beta/v#Adam/sequential_1/conv2d_1/kernel/v!Adam/sequential_1/conv2d_1/bias/v/Adam/sequential_1/batch_normalization_1/gamma/v.Adam/sequential_1/batch_normalization_1/beta/v#Adam/sequential_3/conv2d_3/kernel/v!Adam/sequential_3/conv2d_3/bias/v/Adam/sequential_3/batch_normalization_3/gamma/v.Adam/sequential_3/batch_normalization_3/beta/v#Adam/sequential_5/conv2d_5/kernel/v!Adam/sequential_5/conv2d_5/bias/v/Adam/sequential_5/batch_normalization_5/gamma/v.Adam/sequential_5/batch_normalization_5/beta/v Adam/sequential_6/dense/kernel/vAdam/sequential_6/dense/bias/v*S
TinL
J2H*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_restore_28385Ý÷

ù
,__inference_functional_1_layer_call_fn_26167
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_261122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

»
*__inference_sequential_layer_call_fn_26790

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_244122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_25240

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î/

G__inference_sequential_1_layer_call_and_return_conditional_losses_26831

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_1/AssignNewValue¢&batch_normalization_1/AssignNewValue_1°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpÀ
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp®
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_1/BiasAddm
activation_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_2/mul/x
activation_2/mulMulactivation_2/mul/x:output:0conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/mulo
activation_2/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
activation_2/Sqrt/xm
activation_2/SqrtSqrtactivation_2/Sqrt/x:output:0*
T0*
_output_shapes
: 2
activation_2/Sqrtm
activation_2/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
activation_2/Pow/y
activation_2/PowPowconv2d_1/BiasAdd:output:0activation_2/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/Powq
activation_2/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2
activation_2/mul_1/x 
activation_2/mul_1Mulactivation_2/mul_1/x:output:0activation_2/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/mul_1
activation_2/addAddV2conv2d_1/BiasAdd:output:0activation_2/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/add
activation_2/mul_2Mulactivation_2/Sqrt:y:0activation_2/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/mul_2
activation_2/TanhTanhactivation_2/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/Tanhq
activation_2/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_2/add_1/x£
activation_2/add_1AddV2activation_2/add_1/x:output:0activation_2/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/add_1
activation_2/mul_3Mulactivation_2/mul:z:0activation_2/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/mul_3¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1î
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_2/mul_3:z:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_1/FusedBatchNormV3
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValue
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1Ô
max_pooling2d/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPoolÌ
IdentityIdentitymax_pooling2d/MaxPool:output:0%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ ::::::2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


,__inference_sequential_6_layer_call_fn_25584
flatten_input
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_255772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ  ::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
'
_user_specified_nameflatten_input
°
c
G__inference_activation_1_layer_call_and_return_conditional_losses_24253

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xe
mulMulmul/x:output:0inputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulU
Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
Sqrt/xF
SqrtSqrtSqrt/x:output:0*
T0*
_output_shapes
: 2
SqrtS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
Pow/ye
PowPowinputsPow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
PowW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2	
mul_1/xl
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1b
addAddV2inputs	mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
addd
mul_2MulSqrt:y:0add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2[
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
TanhW
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
add_1/xo
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1e
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3g
IdentityIdentity	mul_3:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
×

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27699

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ@:::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

½
,__inference_sequential_3_layer_call_fn_27014

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_250832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

Á
*__inference_sequential_layer_call_fn_24391
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_243762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input


N__inference_batch_normalization_layer_call_and_return_conditional_losses_27295

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ïÉ
©
G__inference_functional_1_layer_call_and_return_conditional_losses_26564

inputs4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource:
6sequential_batch_normalization_readvariableop_resource<
8sequential_batch_normalization_readvariableop_1_resourceK
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource8
4sequential_1_conv2d_1_conv2d_readvariableop_resource9
5sequential_1_conv2d_1_biasadd_readvariableop_resource>
:sequential_1_batch_normalization_1_readvariableop_resource@
<sequential_1_batch_normalization_1_readvariableop_1_resourceO
Ksequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource8
4sequential_3_conv2d_3_conv2d_readvariableop_resource9
5sequential_3_conv2d_3_biasadd_readvariableop_resource>
:sequential_3_batch_normalization_3_readvariableop_resource@
<sequential_3_batch_normalization_3_readvariableop_1_resourceO
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceQ
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource8
4sequential_5_conv2d_5_conv2d_readvariableop_resource9
5sequential_5_conv2d_5_biasadd_readvariableop_resource>
:sequential_5_batch_normalization_5_readvariableop_resource@
<sequential_5_batch_normalization_5_readvariableop_1_resourceO
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceQ
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource5
1sequential_6_dense_matmul_readvariableop_resource6
2sequential_6_dense_biasadd_readvariableop_resource
identityË
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpÛ
sequential/conv2d/Conv2DConv2Dinputs/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
sequential/conv2d/Conv2DÂ
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOpÒ
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/conv2d/BiasAdd
sequential/activation_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/activation_1/mul/xÉ
sequential/activation_1/mulMul&sequential/activation_1/mul/x:output:0"sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/mul
sequential/activation_1/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2 
sequential/activation_1/Sqrt/x
sequential/activation_1/SqrtSqrt'sequential/activation_1/Sqrt/x:output:0*
T0*
_output_shapes
: 2
sequential/activation_1/Sqrt
sequential/activation_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
sequential/activation_1/Pow/yÉ
sequential/activation_1/PowPow"sequential/conv2d/BiasAdd:output:0&sequential/activation_1/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/Pow
sequential/activation_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2!
sequential/activation_1/mul_1/xÌ
sequential/activation_1/mul_1Mul(sequential/activation_1/mul_1/x:output:0sequential/activation_1/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/mul_1Æ
sequential/activation_1/addAddV2"sequential/conv2d/BiasAdd:output:0!sequential/activation_1/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/addÄ
sequential/activation_1/mul_2Mul sequential/activation_1/Sqrt:y:0sequential/activation_1/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/mul_2£
sequential/activation_1/TanhTanh!sequential/activation_1/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/Tanh
sequential/activation_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
sequential/activation_1/add_1/xÏ
sequential/activation_1/add_1AddV2(sequential/activation_1/add_1/x:output:0 sequential/activation_1/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/add_1Å
sequential/activation_1/mul_3Mulsequential/activation_1/mul:z:0!sequential/activation_1/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/mul_3Ñ
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential/batch_normalization/ReadVariableOp×
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype021
/sequential/batch_normalization/ReadVariableOp_1
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¡
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3!sequential/activation_1/mul_3:z:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3á
max_pooling2d_3/MaxPoolMaxPool3sequential/batch_normalization/FusedBatchNormV3:y:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool×
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+sequential_1/conv2d_1/Conv2D/ReadVariableOp
sequential_1/conv2d_1/Conv2DConv2D3sequential/batch_normalization/FusedBatchNormV3:y:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
sequential_1/conv2d_1/Conv2DÎ
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpâ
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_1/conv2d_1/BiasAdd
sequential_1/activation_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential_1/activation_2/mul/xÓ
sequential_1/activation_2/mulMul(sequential_1/activation_2/mul/x:output:0&sequential_1/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_1/activation_2/mul
 sequential_1/activation_2/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2"
 sequential_1/activation_2/Sqrt/x
sequential_1/activation_2/SqrtSqrt)sequential_1/activation_2/Sqrt/x:output:0*
T0*
_output_shapes
: 2 
sequential_1/activation_2/Sqrt
sequential_1/activation_2/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2!
sequential_1/activation_2/Pow/yÓ
sequential_1/activation_2/PowPow&sequential_1/conv2d_1/BiasAdd:output:0(sequential_1/activation_2/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_1/activation_2/Pow
!sequential_1/activation_2/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2#
!sequential_1/activation_2/mul_1/xÔ
sequential_1/activation_2/mul_1Mul*sequential_1/activation_2/mul_1/x:output:0!sequential_1/activation_2/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential_1/activation_2/mul_1Ð
sequential_1/activation_2/addAddV2&sequential_1/conv2d_1/BiasAdd:output:0#sequential_1/activation_2/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_1/activation_2/addÌ
sequential_1/activation_2/mul_2Mul"sequential_1/activation_2/Sqrt:y:0!sequential_1/activation_2/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential_1/activation_2/mul_2©
sequential_1/activation_2/TanhTanh#sequential_1/activation_2/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_1/activation_2/Tanh
!sequential_1/activation_2/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!sequential_1/activation_2/add_1/x×
sequential_1/activation_2/add_1AddV2*sequential_1/activation_2/add_1/x:output:0"sequential_1/activation_2/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential_1/activation_2/add_1Í
sequential_1/activation_2/mul_3Mul!sequential_1/activation_2/mul:z:0#sequential_1/activation_2/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential_1/activation_2/mul_3Ý
1sequential_1/batch_normalization_1/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype023
1sequential_1/batch_normalization_1/ReadVariableOpã
3sequential_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype025
3sequential_1/batch_normalization_1/ReadVariableOp_1
Bsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1»
3sequential_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#sequential_1/activation_2/mul_3:z:09sequential_1/batch_normalization_1/ReadVariableOp:value:0;sequential_1/batch_normalization_1/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 25
3sequential_1/batch_normalization_1/FusedBatchNormV3û
"sequential_1/max_pooling2d/MaxPoolMaxPool7sequential_1/batch_normalization_1/FusedBatchNormV3:y:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2$
"sequential_1/max_pooling2d/MaxPoolt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisê
concatenate/concatConcatV2 max_pooling2d_3/MaxPool:output:0+sequential_1/max_pooling2d/MaxPool:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
concatenate/concatÇ
max_pooling2d_4/MaxPoolMaxPoolconcatenate/concat:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool×
+sequential_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+sequential_3/conv2d_3/Conv2D/ReadVariableOpü
sequential_3/conv2d_3/Conv2DConv2Dconcatenate/concat:output:03sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
sequential_3/conv2d_3/Conv2DÎ
,sequential_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_3/conv2d_3/BiasAdd/ReadVariableOpâ
sequential_3/conv2d_3/BiasAddBiasAdd%sequential_3/conv2d_3/Conv2D:output:04sequential_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_3/conv2d_3/BiasAdd
sequential_3/activation_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential_3/activation_4/mul/xÓ
sequential_3/activation_4/mulMul(sequential_3/activation_4/mul/x:output:0&sequential_3/conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_3/activation_4/mul
 sequential_3/activation_4/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2"
 sequential_3/activation_4/Sqrt/x
sequential_3/activation_4/SqrtSqrt)sequential_3/activation_4/Sqrt/x:output:0*
T0*
_output_shapes
: 2 
sequential_3/activation_4/Sqrt
sequential_3/activation_4/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2!
sequential_3/activation_4/Pow/yÓ
sequential_3/activation_4/PowPow&sequential_3/conv2d_3/BiasAdd:output:0(sequential_3/activation_4/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_3/activation_4/Pow
!sequential_3/activation_4/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2#
!sequential_3/activation_4/mul_1/xÔ
sequential_3/activation_4/mul_1Mul*sequential_3/activation_4/mul_1/x:output:0!sequential_3/activation_4/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_3/activation_4/mul_1Ð
sequential_3/activation_4/addAddV2&sequential_3/conv2d_3/BiasAdd:output:0#sequential_3/activation_4/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_3/activation_4/addÌ
sequential_3/activation_4/mul_2Mul"sequential_3/activation_4/Sqrt:y:0!sequential_3/activation_4/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_3/activation_4/mul_2©
sequential_3/activation_4/TanhTanh#sequential_3/activation_4/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
sequential_3/activation_4/Tanh
!sequential_3/activation_4/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!sequential_3/activation_4/add_1/x×
sequential_3/activation_4/add_1AddV2*sequential_3/activation_4/add_1/x:output:0"sequential_3/activation_4/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_3/activation_4/add_1Í
sequential_3/activation_4/mul_3Mul!sequential_3/activation_4/mul:z:0#sequential_3/activation_4/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_3/activation_4/mul_3Ý
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOpã
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1»
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3#sequential_3/activation_4/mul_3:z:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 25
3sequential_3/batch_normalization_3/FusedBatchNormV3ý
$sequential_3/max_pooling2d_1/MaxPoolMaxPool7sequential_3/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_1/MaxPoolx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisñ
concatenate_1/concatConcatV2 max_pooling2d_4/MaxPool:output:0-sequential_3/max_pooling2d_1/MaxPool:output:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
concatenate_1/concatÊ
max_pooling2d_5/MaxPoolMaxPoolconcatenate_1/concat:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPoolÙ
+sequential_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_5_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+sequential_5/conv2d_5/Conv2D/ReadVariableOpý
sequential_5/conv2d_5/Conv2DConv2Dconcatenate_1/concat:output:03sequential_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
2
sequential_5/conv2d_5/Conv2DÏ
,sequential_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,sequential_5/conv2d_5/BiasAdd/ReadVariableOpá
sequential_5/conv2d_5/BiasAddBiasAdd%sequential_5/conv2d_5/Conv2D:output:04sequential_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_5/conv2d_5/BiasAdd
sequential_5/activation_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential_5/activation_6/mul/xÒ
sequential_5/activation_6/mulMul(sequential_5/activation_6/mul/x:output:0&sequential_5/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_5/activation_6/mul
 sequential_5/activation_6/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2"
 sequential_5/activation_6/Sqrt/x
sequential_5/activation_6/SqrtSqrt)sequential_5/activation_6/Sqrt/x:output:0*
T0*
_output_shapes
: 2 
sequential_5/activation_6/Sqrt
sequential_5/activation_6/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2!
sequential_5/activation_6/Pow/yÒ
sequential_5/activation_6/PowPow&sequential_5/conv2d_5/BiasAdd:output:0(sequential_5/activation_6/Pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_5/activation_6/Pow
!sequential_5/activation_6/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2#
!sequential_5/activation_6/mul_1/xÓ
sequential_5/activation_6/mul_1Mul*sequential_5/activation_6/mul_1/x:output:0!sequential_5/activation_6/Pow:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_5/activation_6/mul_1Ï
sequential_5/activation_6/addAddV2&sequential_5/conv2d_5/BiasAdd:output:0#sequential_5/activation_6/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_5/activation_6/addË
sequential_5/activation_6/mul_2Mul"sequential_5/activation_6/Sqrt:y:0!sequential_5/activation_6/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_5/activation_6/mul_2¨
sequential_5/activation_6/TanhTanh#sequential_5/activation_6/mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2 
sequential_5/activation_6/Tanh
!sequential_5/activation_6/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!sequential_5/activation_6/add_1/xÖ
sequential_5/activation_6/add_1AddV2*sequential_5/activation_6/add_1/x:output:0"sequential_5/activation_6/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_5/activation_6/add_1Ì
sequential_5/activation_6/mul_3Mul!sequential_5/activation_6/mul:z:0#sequential_5/activation_6/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_5/activation_6/mul_3Þ
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype023
1sequential_5/batch_normalization_5/ReadVariableOpä
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3sequential_5/batch_normalization_5/ReadVariableOp_1
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1¾
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3#sequential_5/activation_6/mul_3:z:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
is_training( 25
3sequential_5/batch_normalization_5/FusedBatchNormV3þ
$sequential_5/max_pooling2d_2/MaxPoolMaxPool7sequential_5/batch_normalization_5/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
2&
$sequential_5/max_pooling2d_2/MaxPoolx
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axisñ
concatenate_2/concatConcatV2 max_pooling2d_5/MaxPool:output:0-sequential_5/max_pooling2d_2/MaxPool:output:0"concatenate_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
concatenate_2/concat
sequential_6/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
sequential_6/flatten/Const¿
sequential_6/flatten/ReshapeReshapeconcatenate_2/concat:output:0#sequential_6/flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/flatten/ReshapeÈ
(sequential_6/dense/MatMul/ReadVariableOpReadVariableOp1sequential_6_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(sequential_6/dense/MatMul/ReadVariableOpË
sequential_6/dense/MatMulMatMul%sequential_6/flatten/Reshape:output:00sequential_6/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/dense/MatMulÅ
)sequential_6/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_6_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_6/dense/BiasAdd/ReadVariableOpÍ
sequential_6/dense/BiasAddBiasAdd#sequential_6/dense/MatMul:product:01sequential_6/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/dense/BiasAdd
sequential_6/dense/SoftmaxSoftmax#sequential_6/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/dense/Softmaxx
IdentityIdentity$sequential_6/dense/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
í
G__inference_sequential_3_layer_call_and_return_conditional_losses_25060
conv2d_3_input
conv2d_3_25043
conv2d_3_25045
batch_normalization_3_25049
batch_normalization_3_25051
batch_normalization_3_25053
batch_normalization_3_25055
identity¢-batch_normalization_3/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¦
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_3_25043conv2d_3_25045*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_249232"
 conv2d_3/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_249572
activation_4/PartitionedCall¼
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0batch_normalization_3_25049batch_normalization_3_25051batch_normalization_3_25053batch_normalization_3_25055*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_250022/
-batch_normalization_3/StatefulPartitionedCall£
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_249032!
max_pooling2d_1/PartitionedCall×
IdentityIdentity(max_pooling2d_1/PartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_3_input

f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_24903

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
H
,__inference_activation_2_layer_call_fn_27427

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_246032
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ø

,__inference_sequential_6_layer_call_fn_27206

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_255582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ  ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
½7

G__inference_functional_1_layer_call_and_return_conditional_losses_25988

inputs
sequential_25924
sequential_25926
sequential_25928
sequential_25930
sequential_25932
sequential_25934
sequential_1_25938
sequential_1_25940
sequential_1_25942
sequential_1_25944
sequential_1_25946
sequential_1_25948
sequential_3_25953
sequential_3_25955
sequential_3_25957
sequential_3_25959
sequential_3_25961
sequential_3_25963
sequential_5_25968
sequential_5_25970
sequential_5_25972
sequential_5_25974
sequential_5_25976
sequential_5_25978
sequential_6_25982
sequential_6_25984
identity¢"sequential/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall¢$sequential_5/StatefulPartitionedCall¢$sequential_6/StatefulPartitionedCallö
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_25924sequential_25926sequential_25928sequential_25930sequential_25932sequential_25934*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_243762$
"sequential/StatefulPartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_244332!
max_pooling2d_3/PartitionedCall­
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_25938sequential_1_25940sequential_1_25942sequential_1_25944sequential_1_25946sequential_1_25948*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_247292&
$sequential_1/StatefulPartitionedCall»
concatenate/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_256902
concatenate/PartitionedCall
max_pooling2d_4/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_247872!
max_pooling2d_4/PartitionedCall¤
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sequential_3_25953sequential_3_25955sequential_3_25957sequential_3_25959sequential_3_25961sequential_3_25963*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_250832&
$sequential_3/StatefulPartitionedCallÀ
concatenate_1/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0-sequential_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_257542
concatenate_1/PartitionedCall
max_pooling2d_5/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_251412!
max_pooling2d_5/PartitionedCall§
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0sequential_5_25968sequential_5_25970sequential_5_25972sequential_5_25974sequential_5_25976sequential_5_25978*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_254372&
$sequential_5/StatefulPartitionedCallÀ
concatenate_2/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0-sequential_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_258182
concatenate_2/PartitionedCallÈ
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0sequential_6_25982sequential_6_25984*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_255582&
$sequential_6/StatefulPartitionedCallÂ
IdentityIdentity-sequential_6/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ù
,__inference_functional_1_layer_call_fn_26043
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity¢StatefulPartitionedCallº
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_259882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Â
»
G__inference_sequential_6_layer_call_and_return_conditional_losses_27184

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape¡
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Softmaxk
IdentityIdentitydense/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ  :::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
ëñ
Õ
 __inference__wrapped_model_24101
input_1A
=functional_1_sequential_conv2d_conv2d_readvariableop_resourceB
>functional_1_sequential_conv2d_biasadd_readvariableop_resourceG
Cfunctional_1_sequential_batch_normalization_readvariableop_resourceI
Efunctional_1_sequential_batch_normalization_readvariableop_1_resourceX
Tfunctional_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceZ
Vfunctional_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceE
Afunctional_1_sequential_1_conv2d_1_conv2d_readvariableop_resourceF
Bfunctional_1_sequential_1_conv2d_1_biasadd_readvariableop_resourceK
Gfunctional_1_sequential_1_batch_normalization_1_readvariableop_resourceM
Ifunctional_1_sequential_1_batch_normalization_1_readvariableop_1_resource\
Xfunctional_1_sequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource^
Zfunctional_1_sequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceE
Afunctional_1_sequential_3_conv2d_3_conv2d_readvariableop_resourceF
Bfunctional_1_sequential_3_conv2d_3_biasadd_readvariableop_resourceK
Gfunctional_1_sequential_3_batch_normalization_3_readvariableop_resourceM
Ifunctional_1_sequential_3_batch_normalization_3_readvariableop_1_resource\
Xfunctional_1_sequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource^
Zfunctional_1_sequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceE
Afunctional_1_sequential_5_conv2d_5_conv2d_readvariableop_resourceF
Bfunctional_1_sequential_5_conv2d_5_biasadd_readvariableop_resourceK
Gfunctional_1_sequential_5_batch_normalization_5_readvariableop_resourceM
Ifunctional_1_sequential_5_batch_normalization_5_readvariableop_1_resource\
Xfunctional_1_sequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource^
Zfunctional_1_sequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceB
>functional_1_sequential_6_dense_matmul_readvariableop_resourceC
?functional_1_sequential_6_dense_biasadd_readvariableop_resource
identityò
4functional_1/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp=functional_1_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype026
4functional_1/sequential/conv2d/Conv2D/ReadVariableOp
%functional_1/sequential/conv2d/Conv2DConv2Dinput_1<functional_1/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2'
%functional_1/sequential/conv2d/Conv2Dé
5functional_1/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp>functional_1_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype027
5functional_1/sequential/conv2d/BiasAdd/ReadVariableOp
&functional_1/sequential/conv2d/BiasAddBiasAdd.functional_1/sequential/conv2d/Conv2D:output:0=functional_1/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2(
&functional_1/sequential/conv2d/BiasAdd
*functional_1/sequential/activation_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2,
*functional_1/sequential/activation_1/mul/xý
(functional_1/sequential/activation_1/mulMul3functional_1/sequential/activation_1/mul/x:output:0/functional_1/sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(functional_1/sequential/activation_1/mul
+functional_1/sequential/activation_1/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2-
+functional_1/sequential/activation_1/Sqrt/xµ
)functional_1/sequential/activation_1/SqrtSqrt4functional_1/sequential/activation_1/Sqrt/x:output:0*
T0*
_output_shapes
: 2+
)functional_1/sequential/activation_1/Sqrt
*functional_1/sequential/activation_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2,
*functional_1/sequential/activation_1/Pow/yý
(functional_1/sequential/activation_1/PowPow/functional_1/sequential/conv2d/BiasAdd:output:03functional_1/sequential/activation_1/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(functional_1/sequential/activation_1/Pow¡
,functional_1/sequential/activation_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2.
,functional_1/sequential/activation_1/mul_1/x
*functional_1/sequential/activation_1/mul_1Mul5functional_1/sequential/activation_1/mul_1/x:output:0,functional_1/sequential/activation_1/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*functional_1/sequential/activation_1/mul_1ú
(functional_1/sequential/activation_1/addAddV2/functional_1/sequential/conv2d/BiasAdd:output:0.functional_1/sequential/activation_1/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2*
(functional_1/sequential/activation_1/addø
*functional_1/sequential/activation_1/mul_2Mul-functional_1/sequential/activation_1/Sqrt:y:0,functional_1/sequential/activation_1/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*functional_1/sequential/activation_1/mul_2Ê
)functional_1/sequential/activation_1/TanhTanh.functional_1/sequential/activation_1/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2+
)functional_1/sequential/activation_1/Tanh¡
,functional_1/sequential/activation_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,functional_1/sequential/activation_1/add_1/x
*functional_1/sequential/activation_1/add_1AddV25functional_1/sequential/activation_1/add_1/x:output:0-functional_1/sequential/activation_1/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*functional_1/sequential/activation_1/add_1ù
*functional_1/sequential/activation_1/mul_3Mul,functional_1/sequential/activation_1/mul:z:0.functional_1/sequential/activation_1/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*functional_1/sequential/activation_1/mul_3ø
:functional_1/sequential/batch_normalization/ReadVariableOpReadVariableOpCfunctional_1_sequential_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02<
:functional_1/sequential/batch_normalization/ReadVariableOpþ
<functional_1/sequential/batch_normalization/ReadVariableOp_1ReadVariableOpEfunctional_1_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02>
<functional_1/sequential/batch_normalization/ReadVariableOp_1«
Kfunctional_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpTfunctional_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02M
Kfunctional_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp±
Mfunctional_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVfunctional_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02O
Mfunctional_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ü
<functional_1/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3.functional_1/sequential/activation_1/mul_3:z:0Bfunctional_1/sequential/batch_normalization/ReadVariableOp:value:0Dfunctional_1/sequential/batch_normalization/ReadVariableOp_1:value:0Sfunctional_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ufunctional_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2>
<functional_1/sequential/batch_normalization/FusedBatchNormV3
$functional_1/max_pooling2d_3/MaxPoolMaxPool@functional_1/sequential/batch_normalization/FusedBatchNormV3:y:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_3/MaxPoolþ
8functional_1/sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpAfunctional_1_sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02:
8functional_1/sequential_1/conv2d_1/Conv2D/ReadVariableOpÈ
)functional_1/sequential_1/conv2d_1/Conv2DConv2D@functional_1/sequential/batch_normalization/FusedBatchNormV3:y:0@functional_1/sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2+
)functional_1/sequential_1/conv2d_1/Conv2Dõ
9functional_1/sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpBfunctional_1_sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9functional_1/sequential_1/conv2d_1/BiasAdd/ReadVariableOp
*functional_1/sequential_1/conv2d_1/BiasAddBiasAdd2functional_1/sequential_1/conv2d_1/Conv2D:output:0Afunctional_1/sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*functional_1/sequential_1/conv2d_1/BiasAdd¡
,functional_1/sequential_1/activation_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,functional_1/sequential_1/activation_2/mul/x
*functional_1/sequential_1/activation_2/mulMul5functional_1/sequential_1/activation_2/mul/x:output:03functional_1/sequential_1/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*functional_1/sequential_1/activation_2/mul£
-functional_1/sequential_1/activation_2/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2/
-functional_1/sequential_1/activation_2/Sqrt/x»
+functional_1/sequential_1/activation_2/SqrtSqrt6functional_1/sequential_1/activation_2/Sqrt/x:output:0*
T0*
_output_shapes
: 2-
+functional_1/sequential_1/activation_2/Sqrt¡
,functional_1/sequential_1/activation_2/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2.
,functional_1/sequential_1/activation_2/Pow/y
*functional_1/sequential_1/activation_2/PowPow3functional_1/sequential_1/conv2d_1/BiasAdd:output:05functional_1/sequential_1/activation_2/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*functional_1/sequential_1/activation_2/Pow¥
.functional_1/sequential_1/activation_2/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=20
.functional_1/sequential_1/activation_2/mul_1/x
,functional_1/sequential_1/activation_2/mul_1Mul7functional_1/sequential_1/activation_2/mul_1/x:output:0.functional_1/sequential_1/activation_2/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,functional_1/sequential_1/activation_2/mul_1
*functional_1/sequential_1/activation_2/addAddV23functional_1/sequential_1/conv2d_1/BiasAdd:output:00functional_1/sequential_1/activation_2/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2,
*functional_1/sequential_1/activation_2/add
,functional_1/sequential_1/activation_2/mul_2Mul/functional_1/sequential_1/activation_2/Sqrt:y:0.functional_1/sequential_1/activation_2/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,functional_1/sequential_1/activation_2/mul_2Ð
+functional_1/sequential_1/activation_2/TanhTanh0functional_1/sequential_1/activation_2/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2-
+functional_1/sequential_1/activation_2/Tanh¥
.functional_1/sequential_1/activation_2/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.functional_1/sequential_1/activation_2/add_1/x
,functional_1/sequential_1/activation_2/add_1AddV27functional_1/sequential_1/activation_2/add_1/x:output:0/functional_1/sequential_1/activation_2/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,functional_1/sequential_1/activation_2/add_1
,functional_1/sequential_1/activation_2/mul_3Mul.functional_1/sequential_1/activation_2/mul:z:00functional_1/sequential_1/activation_2/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2.
,functional_1/sequential_1/activation_2/mul_3
>functional_1/sequential_1/batch_normalization_1/ReadVariableOpReadVariableOpGfunctional_1_sequential_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02@
>functional_1/sequential_1/batch_normalization_1/ReadVariableOp
@functional_1/sequential_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpIfunctional_1_sequential_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@functional_1/sequential_1/batch_normalization_1/ReadVariableOp_1·
Ofunctional_1/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpXfunctional_1_sequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02Q
Ofunctional_1/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp½
Qfunctional_1/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZfunctional_1_sequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02S
Qfunctional_1/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1
@functional_1/sequential_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV30functional_1/sequential_1/activation_2/mul_3:z:0Ffunctional_1/sequential_1/batch_normalization_1/ReadVariableOp:value:0Hfunctional_1/sequential_1/batch_normalization_1/ReadVariableOp_1:value:0Wfunctional_1/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Yfunctional_1/sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2B
@functional_1/sequential_1/batch_normalization_1/FusedBatchNormV3¢
/functional_1/sequential_1/max_pooling2d/MaxPoolMaxPoolDfunctional_1/sequential_1/batch_normalization_1/FusedBatchNormV3:y:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
21
/functional_1/sequential_1/max_pooling2d/MaxPool
$functional_1/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$functional_1/concatenate/concat/axis«
functional_1/concatenate/concatConcatV2-functional_1/max_pooling2d_3/MaxPool:output:08functional_1/sequential_1/max_pooling2d/MaxPool:output:0-functional_1/concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
functional_1/concatenate/concatî
$functional_1/max_pooling2d_4/MaxPoolMaxPool(functional_1/concatenate/concat:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_4/MaxPoolþ
8functional_1/sequential_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpAfunctional_1_sequential_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02:
8functional_1/sequential_3/conv2d_3/Conv2D/ReadVariableOp°
)functional_1/sequential_3/conv2d_3/Conv2DConv2D(functional_1/concatenate/concat:output:0@functional_1/sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2+
)functional_1/sequential_3/conv2d_3/Conv2Dõ
9functional_1/sequential_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpBfunctional_1_sequential_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9functional_1/sequential_3/conv2d_3/BiasAdd/ReadVariableOp
*functional_1/sequential_3/conv2d_3/BiasAddBiasAdd2functional_1/sequential_3/conv2d_3/Conv2D:output:0Afunctional_1/sequential_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*functional_1/sequential_3/conv2d_3/BiasAdd¡
,functional_1/sequential_3/activation_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,functional_1/sequential_3/activation_4/mul/x
*functional_1/sequential_3/activation_4/mulMul5functional_1/sequential_3/activation_4/mul/x:output:03functional_1/sequential_3/conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*functional_1/sequential_3/activation_4/mul£
-functional_1/sequential_3/activation_4/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2/
-functional_1/sequential_3/activation_4/Sqrt/x»
+functional_1/sequential_3/activation_4/SqrtSqrt6functional_1/sequential_3/activation_4/Sqrt/x:output:0*
T0*
_output_shapes
: 2-
+functional_1/sequential_3/activation_4/Sqrt¡
,functional_1/sequential_3/activation_4/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2.
,functional_1/sequential_3/activation_4/Pow/y
*functional_1/sequential_3/activation_4/PowPow3functional_1/sequential_3/conv2d_3/BiasAdd:output:05functional_1/sequential_3/activation_4/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*functional_1/sequential_3/activation_4/Pow¥
.functional_1/sequential_3/activation_4/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=20
.functional_1/sequential_3/activation_4/mul_1/x
,functional_1/sequential_3/activation_4/mul_1Mul7functional_1/sequential_3/activation_4/mul_1/x:output:0.functional_1/sequential_3/activation_4/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,functional_1/sequential_3/activation_4/mul_1
*functional_1/sequential_3/activation_4/addAddV23functional_1/sequential_3/conv2d_3/BiasAdd:output:00functional_1/sequential_3/activation_4/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*functional_1/sequential_3/activation_4/add
,functional_1/sequential_3/activation_4/mul_2Mul/functional_1/sequential_3/activation_4/Sqrt:y:0.functional_1/sequential_3/activation_4/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,functional_1/sequential_3/activation_4/mul_2Ð
+functional_1/sequential_3/activation_4/TanhTanh0functional_1/sequential_3/activation_4/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2-
+functional_1/sequential_3/activation_4/Tanh¥
.functional_1/sequential_3/activation_4/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.functional_1/sequential_3/activation_4/add_1/x
,functional_1/sequential_3/activation_4/add_1AddV27functional_1/sequential_3/activation_4/add_1/x:output:0/functional_1/sequential_3/activation_4/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,functional_1/sequential_3/activation_4/add_1
,functional_1/sequential_3/activation_4/mul_3Mul.functional_1/sequential_3/activation_4/mul:z:00functional_1/sequential_3/activation_4/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,functional_1/sequential_3/activation_4/mul_3
>functional_1/sequential_3/batch_normalization_3/ReadVariableOpReadVariableOpGfunctional_1_sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02@
>functional_1/sequential_3/batch_normalization_3/ReadVariableOp
@functional_1/sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpIfunctional_1_sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@functional_1/sequential_3/batch_normalization_3/ReadVariableOp_1·
Ofunctional_1/sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpXfunctional_1_sequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02Q
Ofunctional_1/sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp½
Qfunctional_1/sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZfunctional_1_sequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02S
Qfunctional_1/sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1
@functional_1/sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV30functional_1/sequential_3/activation_4/mul_3:z:0Ffunctional_1/sequential_3/batch_normalization_3/ReadVariableOp:value:0Hfunctional_1/sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Wfunctional_1/sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Yfunctional_1/sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2B
@functional_1/sequential_3/batch_normalization_3/FusedBatchNormV3¤
1functional_1/sequential_3/max_pooling2d_1/MaxPoolMaxPoolDfunctional_1/sequential_3/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
ksize
*
paddingVALID*
strides
23
1functional_1/sequential_3/max_pooling2d_1/MaxPool
&functional_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&functional_1/concatenate_1/concat/axis²
!functional_1/concatenate_1/concatConcatV2-functional_1/max_pooling2d_4/MaxPool:output:0:functional_1/sequential_3/max_pooling2d_1/MaxPool:output:0/functional_1/concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2#
!functional_1/concatenate_1/concatñ
$functional_1/max_pooling2d_5/MaxPoolMaxPool*functional_1/concatenate_1/concat:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
2&
$functional_1/max_pooling2d_5/MaxPool
8functional_1/sequential_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOpAfunctional_1_sequential_5_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02:
8functional_1/sequential_5/conv2d_5/Conv2D/ReadVariableOp±
)functional_1/sequential_5/conv2d_5/Conv2DConv2D*functional_1/concatenate_1/concat:output:0@functional_1/sequential_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
2+
)functional_1/sequential_5/conv2d_5/Conv2Dö
9functional_1/sequential_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpBfunctional_1_sequential_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02;
9functional_1/sequential_5/conv2d_5/BiasAdd/ReadVariableOp
*functional_1/sequential_5/conv2d_5/BiasAddBiasAdd2functional_1/sequential_5/conv2d_5/Conv2D:output:0Afunctional_1/sequential_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2,
*functional_1/sequential_5/conv2d_5/BiasAdd¡
,functional_1/sequential_5/activation_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,functional_1/sequential_5/activation_6/mul/x
*functional_1/sequential_5/activation_6/mulMul5functional_1/sequential_5/activation_6/mul/x:output:03functional_1/sequential_5/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2,
*functional_1/sequential_5/activation_6/mul£
-functional_1/sequential_5/activation_6/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2/
-functional_1/sequential_5/activation_6/Sqrt/x»
+functional_1/sequential_5/activation_6/SqrtSqrt6functional_1/sequential_5/activation_6/Sqrt/x:output:0*
T0*
_output_shapes
: 2-
+functional_1/sequential_5/activation_6/Sqrt¡
,functional_1/sequential_5/activation_6/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2.
,functional_1/sequential_5/activation_6/Pow/y
*functional_1/sequential_5/activation_6/PowPow3functional_1/sequential_5/conv2d_5/BiasAdd:output:05functional_1/sequential_5/activation_6/Pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2,
*functional_1/sequential_5/activation_6/Pow¥
.functional_1/sequential_5/activation_6/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=20
.functional_1/sequential_5/activation_6/mul_1/x
,functional_1/sequential_5/activation_6/mul_1Mul7functional_1/sequential_5/activation_6/mul_1/x:output:0.functional_1/sequential_5/activation_6/Pow:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2.
,functional_1/sequential_5/activation_6/mul_1
*functional_1/sequential_5/activation_6/addAddV23functional_1/sequential_5/conv2d_5/BiasAdd:output:00functional_1/sequential_5/activation_6/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2,
*functional_1/sequential_5/activation_6/addÿ
,functional_1/sequential_5/activation_6/mul_2Mul/functional_1/sequential_5/activation_6/Sqrt:y:0.functional_1/sequential_5/activation_6/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2.
,functional_1/sequential_5/activation_6/mul_2Ï
+functional_1/sequential_5/activation_6/TanhTanh0functional_1/sequential_5/activation_6/mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2-
+functional_1/sequential_5/activation_6/Tanh¥
.functional_1/sequential_5/activation_6/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.functional_1/sequential_5/activation_6/add_1/x
,functional_1/sequential_5/activation_6/add_1AddV27functional_1/sequential_5/activation_6/add_1/x:output:0/functional_1/sequential_5/activation_6/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2.
,functional_1/sequential_5/activation_6/add_1
,functional_1/sequential_5/activation_6/mul_3Mul.functional_1/sequential_5/activation_6/mul:z:00functional_1/sequential_5/activation_6/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2.
,functional_1/sequential_5/activation_6/mul_3
>functional_1/sequential_5/batch_normalization_5/ReadVariableOpReadVariableOpGfunctional_1_sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype02@
>functional_1/sequential_5/batch_normalization_5/ReadVariableOp
@functional_1/sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOpIfunctional_1_sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02B
@functional_1/sequential_5/batch_normalization_5/ReadVariableOp_1¸
Ofunctional_1/sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpXfunctional_1_sequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02Q
Ofunctional_1/sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp¾
Qfunctional_1/sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZfunctional_1_sequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02S
Qfunctional_1/sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1
@functional_1/sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV30functional_1/sequential_5/activation_6/mul_3:z:0Ffunctional_1/sequential_5/batch_normalization_5/ReadVariableOp:value:0Hfunctional_1/sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Wfunctional_1/sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Yfunctional_1/sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
is_training( 2B
@functional_1/sequential_5/batch_normalization_5/FusedBatchNormV3¥
1functional_1/sequential_5/max_pooling2d_2/MaxPoolMaxPoolDfunctional_1/sequential_5/batch_normalization_5/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
23
1functional_1/sequential_5/max_pooling2d_2/MaxPool
&functional_1/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&functional_1/concatenate_2/concat/axis²
!functional_1/concatenate_2/concatConcatV2-functional_1/max_pooling2d_5/MaxPool:output:0:functional_1/sequential_5/max_pooling2d_2/MaxPool:output:0/functional_1/concatenate_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2#
!functional_1/concatenate_2/concat£
'functional_1/sequential_6/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2)
'functional_1/sequential_6/flatten/Constó
)functional_1/sequential_6/flatten/ReshapeReshape*functional_1/concatenate_2/concat:output:00functional_1/sequential_6/flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)functional_1/sequential_6/flatten/Reshapeï
5functional_1/sequential_6/dense/MatMul/ReadVariableOpReadVariableOp>functional_1_sequential_6_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype027
5functional_1/sequential_6/dense/MatMul/ReadVariableOpÿ
&functional_1/sequential_6/dense/MatMulMatMul2functional_1/sequential_6/flatten/Reshape:output:0=functional_1/sequential_6/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&functional_1/sequential_6/dense/MatMulì
6functional_1/sequential_6/dense/BiasAdd/ReadVariableOpReadVariableOp?functional_1_sequential_6_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6functional_1/sequential_6/dense/BiasAdd/ReadVariableOp
'functional_1/sequential_6/dense/BiasAddBiasAdd0functional_1/sequential_6/dense/MatMul:product:0>functional_1/sequential_6/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/sequential_6/dense/BiasAddÁ
'functional_1/sequential_6/dense/SoftmaxSoftmax0functional_1/sequential_6/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'functional_1/sequential_6/dense/Softmax
IdentityIdentity1functional_1/sequential_6/dense/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
Ù
í
G__inference_sequential_5_layer_call_and_return_conditional_losses_25414
conv2d_5_input
conv2d_5_25397
conv2d_5_25399
batch_normalization_5_25403
batch_normalization_5_25405
batch_normalization_5_25407
batch_normalization_5_25409
identity¢-batch_normalization_5/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall¥
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_25397conv2d_5_25399*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_252772"
 conv2d_5/StatefulPartitionedCall
activation_6/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_253112
activation_6/PartitionedCall»
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0batch_normalization_5_25403batch_normalization_5_25405batch_normalization_5_25407batch_normalization_5_25409*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_253562/
-batch_normalization_5/StatefulPartitionedCall¤
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_252572!
max_pooling2d_2/PartitionedCallØ
IdentityIdentity(max_pooling2d_2/PartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
(
_user_specified_nameconv2d_5_input
¾
å
G__inference_sequential_1_layer_call_and_return_conditional_losses_24729

inputs
conv2d_1_24712
conv2d_1_24714
batch_normalization_1_24718
batch_normalization_1_24720
batch_normalization_1_24722
batch_normalization_1_24724
identity¢-batch_normalization_1/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_24712conv2d_1_24714*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_245692"
 conv2d_1/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_246032
activation_2/PartitionedCallº
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0batch_normalization_1_24718batch_normalization_1_24720batch_normalization_1_24722batch_normalization_1_24724*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_246302/
-batch_normalization_1/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_245492
max_pooling2d/PartitionedCall×
IdentityIdentity&max_pooling2d/PartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ ::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
 
Å
,__inference_sequential_5_layer_call_fn_25452
conv2d_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_254372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
(
_user_specified_nameconv2d_5_input
×

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_25002

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ@:::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

­
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_24984

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ú
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Æ
«
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27277

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Æ
«
N__inference_batch_normalization_layer_call_and_return_conditional_losses_24163

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ç&
³
G__inference_sequential_3_layer_call_and_return_conditional_losses_26997

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpÀ
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv2d_3/Conv2D§
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp®
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_3/BiasAddm
activation_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_4/mul/x
activation_4/mulMulactivation_4/mul/x:output:0conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/mulo
activation_4/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
activation_4/Sqrt/xm
activation_4/SqrtSqrtactivation_4/Sqrt/x:output:0*
T0*
_output_shapes
: 2
activation_4/Sqrtm
activation_4/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
activation_4/Pow/y
activation_4/PowPowconv2d_3/BiasAdd:output:0activation_4/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/Powq
activation_4/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2
activation_4/mul_1/x 
activation_4/mul_1Mulactivation_4/mul_1/x:output:0activation_4/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/mul_1
activation_4/addAddV2conv2d_3/BiasAdd:output:0activation_4/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/add
activation_4/mul_2Mulactivation_4/Sqrt:y:0activation_4/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/mul_2
activation_4/TanhTanhactivation_4/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/Tanhq
activation_4/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_4/add_1/x£
activation_4/add_1AddV2activation_4/add_1/x:output:0activation_4/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/add_1
activation_4/mul_3Mulactivation_4/mul:z:0activation_4/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/mul_3¶
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp¼
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1é
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1à
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_4/mul_3:z:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3Ö
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool|
IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@:::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


N__inference_batch_normalization_layer_call_and_return_conditional_losses_24194

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ª
I
-__inference_max_pooling2d_layer_call_fn_24555

inputs
identityì
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_245492
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

­
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27681

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ú
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú
¨
5__inference_batch_normalization_5_layer_call_fn_27882

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¤
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_253382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ@@::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Å7

G__inference_functional_1_layer_call_and_return_conditional_losses_26112

inputs
sequential_26048
sequential_26050
sequential_26052
sequential_26054
sequential_26056
sequential_26058
sequential_1_26062
sequential_1_26064
sequential_1_26066
sequential_1_26068
sequential_1_26070
sequential_1_26072
sequential_3_26077
sequential_3_26079
sequential_3_26081
sequential_3_26083
sequential_3_26085
sequential_3_26087
sequential_5_26092
sequential_5_26094
sequential_5_26096
sequential_5_26098
sequential_5_26100
sequential_5_26102
sequential_6_26106
sequential_6_26108
identity¢"sequential/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall¢$sequential_5/StatefulPartitionedCall¢$sequential_6/StatefulPartitionedCallø
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_26048sequential_26050sequential_26052sequential_26054sequential_26056sequential_26058*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_244122$
"sequential/StatefulPartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_244332!
max_pooling2d_3/PartitionedCall¯
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_26062sequential_1_26064sequential_1_26066sequential_1_26068sequential_1_26070sequential_1_26072*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_247662&
$sequential_1/StatefulPartitionedCall»
concatenate/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_256902
concatenate/PartitionedCall
max_pooling2d_4/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_247872!
max_pooling2d_4/PartitionedCall¦
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sequential_3_26077sequential_3_26079sequential_3_26081sequential_3_26083sequential_3_26085sequential_3_26087*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_251202&
$sequential_3/StatefulPartitionedCallÀ
concatenate_1/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0-sequential_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_257542
concatenate_1/PartitionedCall
max_pooling2d_5/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_251412!
max_pooling2d_5/PartitionedCall©
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0sequential_5_26092sequential_5_26094sequential_5_26096sequential_5_26098sequential_5_26100sequential_5_26102*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_254742&
$sequential_5/StatefulPartitionedCallÀ
concatenate_2/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0-sequential_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_258182
concatenate_2/PartitionedCallÈ
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0sequential_6_26106sequential_6_26108*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_255772&
$sequential_6/StatefulPartitionedCallÂ
IdentityIdentity-sequential_6/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_25257

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
r
F__inference_concatenate_layer_call_and_return_conditional_losses_26911
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1


P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24532

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

»
*__inference_sequential_layer_call_fn_26773

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_243762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
³
¨
@__inference_dense_layer_call_and_return_conditional_losses_27917

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

}
(__inference_conv2d_1_layer_call_fn_27404

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_245692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬
«
C__inference_conv2d_5_layer_call_and_return_conditional_losses_27735

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ@@:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs

­
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27851

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ@@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_24433

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
H
,__inference_activation_1_layer_call_fn_27257

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_242532
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
´

¨
G__inference_sequential_6_layer_call_and_return_conditional_losses_25558

inputs
dense_25552
dense_25554
identity¢dense/StatefulPartitionedCallÕ
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_254992
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_25552dense_25554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_255182
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ  ::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs


P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_24886

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
È
­
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27617

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ä
å
G__inference_sequential_3_layer_call_and_return_conditional_losses_25120

inputs
conv2d_3_25103
conv2d_3_25105
batch_normalization_3_25109
batch_normalization_3_25111
batch_normalization_3_25113
batch_normalization_3_25115
identity¢-batch_normalization_3/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_25103conv2d_3_25105*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_249232"
 conv2d_3/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_249572
activation_4/PartitionedCall¼
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0batch_normalization_3_25109batch_normalization_3_25111batch_normalization_3_25113batch_normalization_3_25115*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_250022/
-batch_normalization_3/StatefulPartitionedCall£
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_249032!
max_pooling2d_1/PartitionedCall×
IdentityIdentity(max_pooling2d_1/PartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú
í
G__inference_sequential_3_layer_call_and_return_conditional_losses_25040
conv2d_3_input
conv2d_3_24934
conv2d_3_24936
batch_normalization_3_25029
batch_normalization_3_25031
batch_normalization_3_25033
batch_normalization_3_25035
identity¢-batch_normalization_3/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¦
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_3_24934conv2d_3_24936*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_249232"
 conv2d_3/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_249572
activation_4/PartitionedCallº
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0batch_normalization_3_25029batch_normalization_3_25031batch_normalization_3_25033batch_normalization_3_25035*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_249842/
-batch_normalization_3/StatefulPartitionedCall£
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_249032!
max_pooling2d_1/PartitionedCall×
IdentityIdentity(max_pooling2d_1/PartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_3_input
È7

G__inference_functional_1_layer_call_and_return_conditional_losses_25918
input_1
sequential_25854
sequential_25856
sequential_25858
sequential_25860
sequential_25862
sequential_25864
sequential_1_25868
sequential_1_25870
sequential_1_25872
sequential_1_25874
sequential_1_25876
sequential_1_25878
sequential_3_25883
sequential_3_25885
sequential_3_25887
sequential_3_25889
sequential_3_25891
sequential_3_25893
sequential_5_25898
sequential_5_25900
sequential_5_25902
sequential_5_25904
sequential_5_25906
sequential_5_25908
sequential_6_25912
sequential_6_25914
identity¢"sequential/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall¢$sequential_5/StatefulPartitionedCall¢$sequential_6/StatefulPartitionedCallù
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_25854sequential_25856sequential_25858sequential_25860sequential_25862sequential_25864*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_244122$
"sequential/StatefulPartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_244332!
max_pooling2d_3/PartitionedCall¯
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_25868sequential_1_25870sequential_1_25872sequential_1_25874sequential_1_25876sequential_1_25878*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_247662&
$sequential_1/StatefulPartitionedCall»
concatenate/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_256902
concatenate/PartitionedCall
max_pooling2d_4/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_247872!
max_pooling2d_4/PartitionedCall¦
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sequential_3_25883sequential_3_25885sequential_3_25887sequential_3_25889sequential_3_25891sequential_3_25893*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_251202&
$sequential_3/StatefulPartitionedCallÀ
concatenate_1/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0-sequential_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_257542
concatenate_1/PartitionedCall
max_pooling2d_5/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_251412!
max_pooling2d_5/PartitionedCall©
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0sequential_5_25898sequential_5_25900sequential_5_25902sequential_5_25904sequential_5_25906sequential_5_25908*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_254742&
$sequential_5/StatefulPartitionedCallÀ
concatenate_2/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0-sequential_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_258182
concatenate_2/PartitionedCallÈ
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0sequential_6_25912sequential_6_25914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_255772&
$sequential_6/StatefulPartitionedCallÂ
IdentityIdentity-sequential_6/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¿
å
G__inference_sequential_5_layer_call_and_return_conditional_losses_25437

inputs
conv2d_5_25420
conv2d_5_25422
batch_normalization_5_25426
batch_normalization_5_25428
batch_normalization_5_25430
batch_normalization_5_25432
identity¢-batch_normalization_5/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_25420conv2d_5_25422*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_252772"
 conv2d_5/StatefulPartitionedCall
activation_6/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_253112
activation_6/PartitionedCall¹
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0batch_normalization_5_25426batch_normalization_5_25428batch_normalization_5_25430batch_normalization_5_25432*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_253382/
-batch_normalization_5/StatefulPartitionedCall¤
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_252572!
max_pooling2d_2/PartitionedCallØ
IdentityIdentity(max_pooling2d_2/PartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
®
«
C__inference_conv2d_3_layer_call_and_return_conditional_losses_24923

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddn
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
þ
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_24549

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

½
,__inference_sequential_5_layer_call_fn_27141

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_254372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
¬
©
A__inference_conv2d_layer_call_and_return_conditional_losses_24219

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddn
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É

¯
G__inference_sequential_6_layer_call_and_return_conditional_losses_25545
flatten_input
dense_25539
dense_25541
identity¢dense/StatefulPartitionedCallÜ
flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_254992
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_25539dense_25541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_255182
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ  ::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:_ [
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
'
_user_specified_nameflatten_input
°
c
G__inference_activation_1_layer_call_and_return_conditional_losses_27252

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xe
mulMulmul/x:output:0inputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulU
Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
Sqrt/xF
SqrtSqrtSqrt/x:output:0*
T0*
_output_shapes
: 2
SqrtS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
Pow/ye
PowPowinputsPow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
PowW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2	
mul_1/xl
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1b
addAddV2inputs	mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
addd
mul_2MulSqrt:y:0add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2[
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
TanhW
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
add_1/xo
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1e
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3g
IdentityIdentity	mul_3:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¢
Å
,__inference_sequential_5_layer_call_fn_25489
conv2d_5_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_254742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
(
_user_specified_nameconv2d_5_input
É

¯
G__inference_sequential_6_layer_call_and_return_conditional_losses_25535
flatten_input
dense_25529
dense_25531
identity¢dense/StatefulPartitionedCallÜ
flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_254992
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_25529dense_25531*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_255182
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ  ::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:_ [
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
'
_user_specified_nameflatten_input
ø

,__inference_sequential_6_layer_call_fn_27215

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_255772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ  ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¦
Å
,__inference_sequential_1_layer_call_fn_24781
conv2d_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_247662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_1_input
î
ð
#__inference_signature_wrapper_26234
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_241012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

ø
,__inference_functional_1_layer_call_fn_26621

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_259882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
K
/__inference_max_pooling2d_1_layer_call_fn_24909

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_249032
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
»
G__inference_sequential_6_layer_call_and_return_conditional_losses_27197

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten/Const
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape¡
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAdds
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Softmaxk
IdentityIdentitydense/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ  :::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
¤
Å
,__inference_sequential_1_layer_call_fn_24744
conv2d_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_247292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_1_input
Ð/

G__inference_sequential_3_layer_call_and_return_conditional_losses_26958

inputs+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_3/AssignNewValue¢&batch_normalization_3/AssignNewValue_1°
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpÀ
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv2d_3/Conv2D§
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp®
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_3/BiasAddm
activation_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_4/mul/x
activation_4/mulMulactivation_4/mul/x:output:0conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/mulo
activation_4/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
activation_4/Sqrt/xm
activation_4/SqrtSqrtactivation_4/Sqrt/x:output:0*
T0*
_output_shapes
: 2
activation_4/Sqrtm
activation_4/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
activation_4/Pow/y
activation_4/PowPowconv2d_3/BiasAdd:output:0activation_4/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/Powq
activation_4/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2
activation_4/mul_1/x 
activation_4/mul_1Mulactivation_4/mul_1/x:output:0activation_4/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/mul_1
activation_4/addAddV2conv2d_3/BiasAdd:output:0activation_4/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/add
activation_4/mul_2Mulactivation_4/Sqrt:y:0activation_4/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/mul_2
activation_4/TanhTanhactivation_4/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/Tanhq
activation_4/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_4/add_1/x£
activation_4/add_1AddV2activation_4/add_1/x:output:0activation_4/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/add_1
activation_4/mul_3Mulactivation_4/mul:z:0activation_4/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_4/mul_3¶
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp¼
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1é
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1î
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_4/mul_3:z:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_3/FusedBatchNormV3
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValue
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1Ö
max_pooling2d_1/MaxPoolMaxPool*batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPoolÌ
IdentityIdentity max_pooling2d_1/MaxPool:output:0%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

Á
*__inference_sequential_layer_call_fn_24427
conv2d_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_244122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
ç
r
H__inference_concatenate_1_layer_call_and_return_conditional_losses_25754

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ@@@:ÿÿÿÿÿÿÿÿÿ@@@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
 
_user_specified_nameinputs

­
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24630

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ú
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
È
­
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27447

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
×

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24648

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ :::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
À7

G__inference_functional_1_layer_call_and_return_conditional_losses_25851
input_1
sequential_25622
sequential_25624
sequential_25626
sequential_25628
sequential_25630
sequential_25632
sequential_1_25670
sequential_1_25672
sequential_1_25674
sequential_1_25676
sequential_1_25678
sequential_1_25680
sequential_3_25734
sequential_3_25736
sequential_3_25738
sequential_3_25740
sequential_3_25742
sequential_3_25744
sequential_5_25798
sequential_5_25800
sequential_5_25802
sequential_5_25804
sequential_5_25806
sequential_5_25808
sequential_6_25845
sequential_6_25847
identity¢"sequential/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall¢$sequential_5/StatefulPartitionedCall¢$sequential_6/StatefulPartitionedCall÷
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_25622sequential_25624sequential_25626sequential_25628sequential_25630sequential_25632*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_243762$
"sequential/StatefulPartitionedCall
max_pooling2d_3/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_244332!
max_pooling2d_3/PartitionedCall­
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_25670sequential_1_25672sequential_1_25674sequential_1_25676sequential_1_25678sequential_1_25680*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_247292&
$sequential_1/StatefulPartitionedCall»
concatenate/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0-sequential_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_256902
concatenate/PartitionedCall
max_pooling2d_4/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_247872!
max_pooling2d_4/PartitionedCall¤
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sequential_3_25734sequential_3_25736sequential_3_25738sequential_3_25740sequential_3_25742sequential_3_25744*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_250832&
$sequential_3/StatefulPartitionedCallÀ
concatenate_1/PartitionedCallPartitionedCall(max_pooling2d_4/PartitionedCall:output:0-sequential_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_257542
concatenate_1/PartitionedCall
max_pooling2d_5/PartitionedCallPartitionedCall&concatenate_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_251412!
max_pooling2d_5/PartitionedCall§
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0sequential_5_25798sequential_5_25800sequential_5_25802sequential_5_25804sequential_5_25806sequential_5_25808*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_254372&
$sequential_5/StatefulPartitionedCallÀ
concatenate_2/PartitionedCallPartitionedCall(max_pooling2d_5/PartitionedCall:output:0-sequential_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_258182
concatenate_2/PartitionedCallÈ
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0sequential_6_25845sequential_6_25847*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_255582&
$sequential_6/StatefulPartitionedCallÂ
IdentityIdentity-sequential_6/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1


,__inference_sequential_6_layer_call_fn_25565
flatten_input
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_6_layer_call_and_return_conditional_losses_255582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ  ::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
'
_user_specified_nameflatten_input


P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27465

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ :::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
®
«
C__inference_conv2d_1_layer_call_and_return_conditional_losses_27395

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddn
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ :::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Þ
¨
5__inference_batch_normalization_3_layer_call_fn_27712

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_249842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27805

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
K
/__inference_max_pooling2d_4_layer_call_fn_24793

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_247872
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
¦
3__inference_batch_normalization_layer_call_fn_27385

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_242982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¤
¨
5__inference_batch_normalization_5_layer_call_fn_27831

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_252402
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

½
,__inference_sequential_3_layer_call_fn_27031

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_251202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ó
t
H__inference_concatenate_2_layer_call_and_return_conditional_losses_27165
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  :Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/1
Þ
¨
5__inference_batch_normalization_1_layer_call_fn_27542

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¥
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_246302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
À
å
G__inference_sequential_1_layer_call_and_return_conditional_losses_24766

inputs
conv2d_1_24749
conv2d_1_24751
batch_normalization_1_24755
batch_normalization_1_24757
batch_normalization_1_24759
batch_normalization_1_24761
identity¢-batch_normalization_1/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_24749conv2d_1_24751*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_245692"
 conv2d_1/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_246032
activation_2/PartitionedCall¼
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0batch_normalization_1_24755batch_normalization_1_24757batch_normalization_1_24759batch_normalization_1_24761*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_246482/
-batch_normalization_1/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_245492
max_pooling2d/PartitionedCall×
IdentityIdentity&max_pooling2d/PartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ ::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¨
5__inference_batch_normalization_3_layer_call_fn_27648

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_248552
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®
K
/__inference_max_pooling2d_2_layer_call_fn_25263

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_252572
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ù
E__inference_sequential_layer_call_and_return_conditional_losses_24335
conv2d_input
conv2d_24230
conv2d_24232
batch_normalization_24325
batch_normalization_24327
batch_normalization_24329
batch_normalization_24331
identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_24230conv2d_24232*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_242192 
conv2d/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_242532
activation_1/PartitionedCall¬
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_24325batch_normalization_24327batch_normalization_24329batch_normalization_24331*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_242802-
+batch_normalization/StatefulPartitionedCallá
IdentityIdentity4batch_normalization/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:_ [
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
 
¨
5__inference_batch_normalization_1_layer_call_fn_27491

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_245322
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¥
c
G__inference_activation_6_layer_call_and_return_conditional_losses_27762

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xd
mulMulmul/x:output:0inputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
mulU
Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
Sqrt/xF
SqrtSqrtSqrt/x:output:0*
T0*
_output_shapes
: 2
SqrtS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
Pow/yd
PowPowinputsPow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
PowW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2	
mul_1/xk
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
mul_1a
addAddV2inputs	mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
addc
mul_2MulSqrt:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
mul_2Z
TanhTanh	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
TanhW
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
add_1/xn
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
add_1d
mul_3Mulmul:z:0	add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
mul_3f
IdentityIdentity	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs

õ%
__inference__traced_save_28162
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop7
3savev2_sequential_conv2d_kernel_read_readvariableop5
1savev2_sequential_conv2d_bias_read_readvariableopC
?savev2_sequential_batch_normalization_gamma_read_readvariableopB
>savev2_sequential_batch_normalization_beta_read_readvariableop;
7savev2_sequential_1_conv2d_1_kernel_read_readvariableop9
5savev2_sequential_1_conv2d_1_bias_read_readvariableopG
Csavev2_sequential_1_batch_normalization_1_gamma_read_readvariableopF
Bsavev2_sequential_1_batch_normalization_1_beta_read_readvariableop;
7savev2_sequential_3_conv2d_3_kernel_read_readvariableop9
5savev2_sequential_3_conv2d_3_bias_read_readvariableopG
Csavev2_sequential_3_batch_normalization_3_gamma_read_readvariableopF
Bsavev2_sequential_3_batch_normalization_3_beta_read_readvariableop;
7savev2_sequential_5_conv2d_5_kernel_read_readvariableop9
5savev2_sequential_5_conv2d_5_bias_read_readvariableopG
Csavev2_sequential_5_batch_normalization_5_gamma_read_readvariableopF
Bsavev2_sequential_5_batch_normalization_5_beta_read_readvariableop8
4savev2_sequential_6_dense_kernel_read_readvariableop6
2savev2_sequential_6_dense_bias_read_readvariableopI
Esavev2_sequential_batch_normalization_moving_mean_read_readvariableopM
Isavev2_sequential_batch_normalization_moving_variance_read_readvariableopM
Isavev2_sequential_1_batch_normalization_1_moving_mean_read_readvariableopQ
Msavev2_sequential_1_batch_normalization_1_moving_variance_read_readvariableopM
Isavev2_sequential_3_batch_normalization_3_moving_mean_read_readvariableopQ
Msavev2_sequential_3_batch_normalization_3_moving_variance_read_readvariableopM
Isavev2_sequential_5_batch_normalization_5_moving_mean_read_readvariableopQ
Msavev2_sequential_5_batch_normalization_5_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop>
:savev2_adam_sequential_conv2d_kernel_m_read_readvariableop<
8savev2_adam_sequential_conv2d_bias_m_read_readvariableopJ
Fsavev2_adam_sequential_batch_normalization_gamma_m_read_readvariableopI
Esavev2_adam_sequential_batch_normalization_beta_m_read_readvariableopB
>savev2_adam_sequential_1_conv2d_1_kernel_m_read_readvariableop@
<savev2_adam_sequential_1_conv2d_1_bias_m_read_readvariableopN
Jsavev2_adam_sequential_1_batch_normalization_1_gamma_m_read_readvariableopM
Isavev2_adam_sequential_1_batch_normalization_1_beta_m_read_readvariableopB
>savev2_adam_sequential_3_conv2d_3_kernel_m_read_readvariableop@
<savev2_adam_sequential_3_conv2d_3_bias_m_read_readvariableopN
Jsavev2_adam_sequential_3_batch_normalization_3_gamma_m_read_readvariableopM
Isavev2_adam_sequential_3_batch_normalization_3_beta_m_read_readvariableopB
>savev2_adam_sequential_5_conv2d_5_kernel_m_read_readvariableop@
<savev2_adam_sequential_5_conv2d_5_bias_m_read_readvariableopN
Jsavev2_adam_sequential_5_batch_normalization_5_gamma_m_read_readvariableopM
Isavev2_adam_sequential_5_batch_normalization_5_beta_m_read_readvariableop?
;savev2_adam_sequential_6_dense_kernel_m_read_readvariableop=
9savev2_adam_sequential_6_dense_bias_m_read_readvariableop>
:savev2_adam_sequential_conv2d_kernel_v_read_readvariableop<
8savev2_adam_sequential_conv2d_bias_v_read_readvariableopJ
Fsavev2_adam_sequential_batch_normalization_gamma_v_read_readvariableopI
Esavev2_adam_sequential_batch_normalization_beta_v_read_readvariableopB
>savev2_adam_sequential_1_conv2d_1_kernel_v_read_readvariableop@
<savev2_adam_sequential_1_conv2d_1_bias_v_read_readvariableopN
Jsavev2_adam_sequential_1_batch_normalization_1_gamma_v_read_readvariableopM
Isavev2_adam_sequential_1_batch_normalization_1_beta_v_read_readvariableopB
>savev2_adam_sequential_3_conv2d_3_kernel_v_read_readvariableop@
<savev2_adam_sequential_3_conv2d_3_bias_v_read_readvariableopN
Jsavev2_adam_sequential_3_batch_normalization_3_gamma_v_read_readvariableopM
Isavev2_adam_sequential_3_batch_normalization_3_beta_v_read_readvariableopB
>savev2_adam_sequential_5_conv2d_5_kernel_v_read_readvariableop@
<savev2_adam_sequential_5_conv2d_5_bias_v_read_readvariableopN
Jsavev2_adam_sequential_5_batch_normalization_5_gamma_v_read_readvariableopM
Isavev2_adam_sequential_5_batch_normalization_5_beta_v_read_readvariableop?
;savev2_adam_sequential_6_dense_kernel_v_read_readvariableop=
9savev2_adam_sequential_6_dense_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b20bad03cdf24c9a84714e402c7d6a18/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¸$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*Ê#
valueÀ#B½#HB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*¥
valueBHB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesï$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop3savev2_sequential_conv2d_kernel_read_readvariableop1savev2_sequential_conv2d_bias_read_readvariableop?savev2_sequential_batch_normalization_gamma_read_readvariableop>savev2_sequential_batch_normalization_beta_read_readvariableop7savev2_sequential_1_conv2d_1_kernel_read_readvariableop5savev2_sequential_1_conv2d_1_bias_read_readvariableopCsavev2_sequential_1_batch_normalization_1_gamma_read_readvariableopBsavev2_sequential_1_batch_normalization_1_beta_read_readvariableop7savev2_sequential_3_conv2d_3_kernel_read_readvariableop5savev2_sequential_3_conv2d_3_bias_read_readvariableopCsavev2_sequential_3_batch_normalization_3_gamma_read_readvariableopBsavev2_sequential_3_batch_normalization_3_beta_read_readvariableop7savev2_sequential_5_conv2d_5_kernel_read_readvariableop5savev2_sequential_5_conv2d_5_bias_read_readvariableopCsavev2_sequential_5_batch_normalization_5_gamma_read_readvariableopBsavev2_sequential_5_batch_normalization_5_beta_read_readvariableop4savev2_sequential_6_dense_kernel_read_readvariableop2savev2_sequential_6_dense_bias_read_readvariableopEsavev2_sequential_batch_normalization_moving_mean_read_readvariableopIsavev2_sequential_batch_normalization_moving_variance_read_readvariableopIsavev2_sequential_1_batch_normalization_1_moving_mean_read_readvariableopMsavev2_sequential_1_batch_normalization_1_moving_variance_read_readvariableopIsavev2_sequential_3_batch_normalization_3_moving_mean_read_readvariableopMsavev2_sequential_3_batch_normalization_3_moving_variance_read_readvariableopIsavev2_sequential_5_batch_normalization_5_moving_mean_read_readvariableopMsavev2_sequential_5_batch_normalization_5_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop:savev2_adam_sequential_conv2d_kernel_m_read_readvariableop8savev2_adam_sequential_conv2d_bias_m_read_readvariableopFsavev2_adam_sequential_batch_normalization_gamma_m_read_readvariableopEsavev2_adam_sequential_batch_normalization_beta_m_read_readvariableop>savev2_adam_sequential_1_conv2d_1_kernel_m_read_readvariableop<savev2_adam_sequential_1_conv2d_1_bias_m_read_readvariableopJsavev2_adam_sequential_1_batch_normalization_1_gamma_m_read_readvariableopIsavev2_adam_sequential_1_batch_normalization_1_beta_m_read_readvariableop>savev2_adam_sequential_3_conv2d_3_kernel_m_read_readvariableop<savev2_adam_sequential_3_conv2d_3_bias_m_read_readvariableopJsavev2_adam_sequential_3_batch_normalization_3_gamma_m_read_readvariableopIsavev2_adam_sequential_3_batch_normalization_3_beta_m_read_readvariableop>savev2_adam_sequential_5_conv2d_5_kernel_m_read_readvariableop<savev2_adam_sequential_5_conv2d_5_bias_m_read_readvariableopJsavev2_adam_sequential_5_batch_normalization_5_gamma_m_read_readvariableopIsavev2_adam_sequential_5_batch_normalization_5_beta_m_read_readvariableop;savev2_adam_sequential_6_dense_kernel_m_read_readvariableop9savev2_adam_sequential_6_dense_bias_m_read_readvariableop:savev2_adam_sequential_conv2d_kernel_v_read_readvariableop8savev2_adam_sequential_conv2d_bias_v_read_readvariableopFsavev2_adam_sequential_batch_normalization_gamma_v_read_readvariableopEsavev2_adam_sequential_batch_normalization_beta_v_read_readvariableop>savev2_adam_sequential_1_conv2d_1_kernel_v_read_readvariableop<savev2_adam_sequential_1_conv2d_1_bias_v_read_readvariableopJsavev2_adam_sequential_1_batch_normalization_1_gamma_v_read_readvariableopIsavev2_adam_sequential_1_batch_normalization_1_beta_v_read_readvariableop>savev2_adam_sequential_3_conv2d_3_kernel_v_read_readvariableop<savev2_adam_sequential_3_conv2d_3_bias_v_read_readvariableopJsavev2_adam_sequential_3_batch_normalization_3_gamma_v_read_readvariableopIsavev2_adam_sequential_3_batch_normalization_3_beta_v_read_readvariableop>savev2_adam_sequential_5_conv2d_5_kernel_v_read_readvariableop<savev2_adam_sequential_5_conv2d_5_bias_v_read_readvariableopJsavev2_adam_sequential_5_batch_normalization_5_gamma_v_read_readvariableopIsavev2_adam_sequential_5_batch_normalization_5_beta_v_read_readvariableop;savev2_adam_sequential_6_dense_kernel_v_read_readvariableop9savev2_adam_sequential_6_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *V
dtypesL
J2H	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ò
_input_shapesÀ
½: : : : : : : : : : :  : : : :@@:@:@:@:::::
:: : : : :@:@::: : : : : : : : :  : : : :@@:@:@:@:::::
:: : : : :  : : : :@@:@:@:@:::::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: :,
(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
: 

_output_shapes
:: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
:@: 

_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
:: 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :,$(
&
_output_shapes
: : %

_output_shapes
: : &

_output_shapes
: : '

_output_shapes
: :,((
&
_output_shapes
:  : )

_output_shapes
: : *

_output_shapes
: : +

_output_shapes
: :,,(
&
_output_shapes
:@@: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@:.0*
(
_output_shapes
::!1

_output_shapes	
::!2

_output_shapes	
::!3

_output_shapes	
::&4"
 
_output_shapes
:
: 5

_output_shapes
::,6(
&
_output_shapes
: : 7

_output_shapes
: : 8

_output_shapes
: : 9

_output_shapes
: :,:(
&
_output_shapes
:  : ;

_output_shapes
: : <

_output_shapes
: : =

_output_shapes
: :,>(
&
_output_shapes
:@@: ?

_output_shapes
:@: @

_output_shapes
:@: A

_output_shapes
:@:.B*
(
_output_shapes
::!C

_output_shapes	
::!D

_output_shapes	
::!E

_output_shapes	
::&F"
 
_output_shapes
:
: G

_output_shapes
::H

_output_shapes
: 

Ù
E__inference_sequential_layer_call_and_return_conditional_losses_24354
conv2d_input
conv2d_24338
conv2d_24340
batch_normalization_24344
batch_normalization_24346
batch_normalization_24348
batch_normalization_24350
identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_24338conv2d_24340*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_242192 
conv2d/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_242532
activation_1/PartitionedCall®
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_24344batch_normalization_24346batch_normalization_24348batch_normalization_24350*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_242982-
+batch_normalization/StatefulPartitionedCallá
IdentityIdentity4batch_normalization/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:_ [
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_nameconv2d_input
®
«
C__inference_conv2d_1_layer_call_and_return_conditional_losses_24569

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddn
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ :::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
þ¶
ã-
!__inference__traced_restore_28385
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate/
+assignvariableop_5_sequential_conv2d_kernel-
)assignvariableop_6_sequential_conv2d_bias;
7assignvariableop_7_sequential_batch_normalization_gamma:
6assignvariableop_8_sequential_batch_normalization_beta3
/assignvariableop_9_sequential_1_conv2d_1_kernel2
.assignvariableop_10_sequential_1_conv2d_1_bias@
<assignvariableop_11_sequential_1_batch_normalization_1_gamma?
;assignvariableop_12_sequential_1_batch_normalization_1_beta4
0assignvariableop_13_sequential_3_conv2d_3_kernel2
.assignvariableop_14_sequential_3_conv2d_3_bias@
<assignvariableop_15_sequential_3_batch_normalization_3_gamma?
;assignvariableop_16_sequential_3_batch_normalization_3_beta4
0assignvariableop_17_sequential_5_conv2d_5_kernel2
.assignvariableop_18_sequential_5_conv2d_5_bias@
<assignvariableop_19_sequential_5_batch_normalization_5_gamma?
;assignvariableop_20_sequential_5_batch_normalization_5_beta1
-assignvariableop_21_sequential_6_dense_kernel/
+assignvariableop_22_sequential_6_dense_biasB
>assignvariableop_23_sequential_batch_normalization_moving_meanF
Bassignvariableop_24_sequential_batch_normalization_moving_varianceF
Bassignvariableop_25_sequential_1_batch_normalization_1_moving_meanJ
Fassignvariableop_26_sequential_1_batch_normalization_1_moving_varianceF
Bassignvariableop_27_sequential_3_batch_normalization_3_moving_meanJ
Fassignvariableop_28_sequential_3_batch_normalization_3_moving_varianceF
Bassignvariableop_29_sequential_5_batch_normalization_5_moving_meanJ
Fassignvariableop_30_sequential_5_batch_normalization_5_moving_variance
assignvariableop_31_total
assignvariableop_32_count
assignvariableop_33_total_1
assignvariableop_34_count_17
3assignvariableop_35_adam_sequential_conv2d_kernel_m5
1assignvariableop_36_adam_sequential_conv2d_bias_mC
?assignvariableop_37_adam_sequential_batch_normalization_gamma_mB
>assignvariableop_38_adam_sequential_batch_normalization_beta_m;
7assignvariableop_39_adam_sequential_1_conv2d_1_kernel_m9
5assignvariableop_40_adam_sequential_1_conv2d_1_bias_mG
Cassignvariableop_41_adam_sequential_1_batch_normalization_1_gamma_mF
Bassignvariableop_42_adam_sequential_1_batch_normalization_1_beta_m;
7assignvariableop_43_adam_sequential_3_conv2d_3_kernel_m9
5assignvariableop_44_adam_sequential_3_conv2d_3_bias_mG
Cassignvariableop_45_adam_sequential_3_batch_normalization_3_gamma_mF
Bassignvariableop_46_adam_sequential_3_batch_normalization_3_beta_m;
7assignvariableop_47_adam_sequential_5_conv2d_5_kernel_m9
5assignvariableop_48_adam_sequential_5_conv2d_5_bias_mG
Cassignvariableop_49_adam_sequential_5_batch_normalization_5_gamma_mF
Bassignvariableop_50_adam_sequential_5_batch_normalization_5_beta_m8
4assignvariableop_51_adam_sequential_6_dense_kernel_m6
2assignvariableop_52_adam_sequential_6_dense_bias_m7
3assignvariableop_53_adam_sequential_conv2d_kernel_v5
1assignvariableop_54_adam_sequential_conv2d_bias_vC
?assignvariableop_55_adam_sequential_batch_normalization_gamma_vB
>assignvariableop_56_adam_sequential_batch_normalization_beta_v;
7assignvariableop_57_adam_sequential_1_conv2d_1_kernel_v9
5assignvariableop_58_adam_sequential_1_conv2d_1_bias_vG
Cassignvariableop_59_adam_sequential_1_batch_normalization_1_gamma_vF
Bassignvariableop_60_adam_sequential_1_batch_normalization_1_beta_v;
7assignvariableop_61_adam_sequential_3_conv2d_3_kernel_v9
5assignvariableop_62_adam_sequential_3_conv2d_3_bias_vG
Cassignvariableop_63_adam_sequential_3_batch_normalization_3_gamma_vF
Bassignvariableop_64_adam_sequential_3_batch_normalization_3_beta_v;
7assignvariableop_65_adam_sequential_5_conv2d_5_kernel_v9
5assignvariableop_66_adam_sequential_5_conv2d_5_bias_vG
Cassignvariableop_67_adam_sequential_5_batch_normalization_5_gamma_vF
Bassignvariableop_68_adam_sequential_5_batch_normalization_5_beta_v8
4assignvariableop_69_adam_sequential_6_dense_kernel_v6
2assignvariableop_70_adam_sequential_6_dense_bias_v
identity_72¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_8¢AssignVariableOp_9¾$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*Ê#
valueÀ#B½#HB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¡
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:H*
dtype0*¥
valueBHB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¶
_output_shapes£
 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*V
dtypesL
J2H	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2£
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ª
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5°
AssignVariableOp_5AssignVariableOp+assignvariableop_5_sequential_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6®
AssignVariableOp_6AssignVariableOp)assignvariableop_6_sequential_conv2d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¼
AssignVariableOp_7AssignVariableOp7assignvariableop_7_sequential_batch_normalization_gammaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8»
AssignVariableOp_8AssignVariableOp6assignvariableop_8_sequential_batch_normalization_betaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9´
AssignVariableOp_9AssignVariableOp/assignvariableop_9_sequential_1_conv2d_1_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¶
AssignVariableOp_10AssignVariableOp.assignvariableop_10_sequential_1_conv2d_1_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ä
AssignVariableOp_11AssignVariableOp<assignvariableop_11_sequential_1_batch_normalization_1_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ã
AssignVariableOp_12AssignVariableOp;assignvariableop_12_sequential_1_batch_normalization_1_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¸
AssignVariableOp_13AssignVariableOp0assignvariableop_13_sequential_3_conv2d_3_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¶
AssignVariableOp_14AssignVariableOp.assignvariableop_14_sequential_3_conv2d_3_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ä
AssignVariableOp_15AssignVariableOp<assignvariableop_15_sequential_3_batch_normalization_3_gammaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ã
AssignVariableOp_16AssignVariableOp;assignvariableop_16_sequential_3_batch_normalization_3_betaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¸
AssignVariableOp_17AssignVariableOp0assignvariableop_17_sequential_5_conv2d_5_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¶
AssignVariableOp_18AssignVariableOp.assignvariableop_18_sequential_5_conv2d_5_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ä
AssignVariableOp_19AssignVariableOp<assignvariableop_19_sequential_5_batch_normalization_5_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ã
AssignVariableOp_20AssignVariableOp;assignvariableop_20_sequential_5_batch_normalization_5_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21µ
AssignVariableOp_21AssignVariableOp-assignvariableop_21_sequential_6_dense_kernelIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22³
AssignVariableOp_22AssignVariableOp+assignvariableop_22_sequential_6_dense_biasIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Æ
AssignVariableOp_23AssignVariableOp>assignvariableop_23_sequential_batch_normalization_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Ê
AssignVariableOp_24AssignVariableOpBassignvariableop_24_sequential_batch_normalization_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ê
AssignVariableOp_25AssignVariableOpBassignvariableop_25_sequential_1_batch_normalization_1_moving_meanIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Î
AssignVariableOp_26AssignVariableOpFassignvariableop_26_sequential_1_batch_normalization_1_moving_varianceIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ê
AssignVariableOp_27AssignVariableOpBassignvariableop_27_sequential_3_batch_normalization_3_moving_meanIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Î
AssignVariableOp_28AssignVariableOpFassignvariableop_28_sequential_3_batch_normalization_3_moving_varianceIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Ê
AssignVariableOp_29AssignVariableOpBassignvariableop_29_sequential_5_batch_normalization_5_moving_meanIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30Î
AssignVariableOp_30AssignVariableOpFassignvariableop_30_sequential_5_batch_normalization_5_moving_varianceIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31¡
AssignVariableOp_31AssignVariableOpassignvariableop_31_totalIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¡
AssignVariableOp_32AssignVariableOpassignvariableop_32_countIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33£
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34£
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35»
AssignVariableOp_35AssignVariableOp3assignvariableop_35_adam_sequential_conv2d_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¹
AssignVariableOp_36AssignVariableOp1assignvariableop_36_adam_sequential_conv2d_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ç
AssignVariableOp_37AssignVariableOp?assignvariableop_37_adam_sequential_batch_normalization_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Æ
AssignVariableOp_38AssignVariableOp>assignvariableop_38_adam_sequential_batch_normalization_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¿
AssignVariableOp_39AssignVariableOp7assignvariableop_39_adam_sequential_1_conv2d_1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40½
AssignVariableOp_40AssignVariableOp5assignvariableop_40_adam_sequential_1_conv2d_1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41Ë
AssignVariableOp_41AssignVariableOpCassignvariableop_41_adam_sequential_1_batch_normalization_1_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ê
AssignVariableOp_42AssignVariableOpBassignvariableop_42_adam_sequential_1_batch_normalization_1_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¿
AssignVariableOp_43AssignVariableOp7assignvariableop_43_adam_sequential_3_conv2d_3_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44½
AssignVariableOp_44AssignVariableOp5assignvariableop_44_adam_sequential_3_conv2d_3_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45Ë
AssignVariableOp_45AssignVariableOpCassignvariableop_45_adam_sequential_3_batch_normalization_3_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46Ê
AssignVariableOp_46AssignVariableOpBassignvariableop_46_adam_sequential_3_batch_normalization_3_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47¿
AssignVariableOp_47AssignVariableOp7assignvariableop_47_adam_sequential_5_conv2d_5_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48½
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_sequential_5_conv2d_5_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ë
AssignVariableOp_49AssignVariableOpCassignvariableop_49_adam_sequential_5_batch_normalization_5_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Ê
AssignVariableOp_50AssignVariableOpBassignvariableop_50_adam_sequential_5_batch_normalization_5_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¼
AssignVariableOp_51AssignVariableOp4assignvariableop_51_adam_sequential_6_dense_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52º
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_sequential_6_dense_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53»
AssignVariableOp_53AssignVariableOp3assignvariableop_53_adam_sequential_conv2d_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54¹
AssignVariableOp_54AssignVariableOp1assignvariableop_54_adam_sequential_conv2d_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55Ç
AssignVariableOp_55AssignVariableOp?assignvariableop_55_adam_sequential_batch_normalization_gamma_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56Æ
AssignVariableOp_56AssignVariableOp>assignvariableop_56_adam_sequential_batch_normalization_beta_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57¿
AssignVariableOp_57AssignVariableOp7assignvariableop_57_adam_sequential_1_conv2d_1_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58½
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_sequential_1_conv2d_1_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59Ë
AssignVariableOp_59AssignVariableOpCassignvariableop_59_adam_sequential_1_batch_normalization_1_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60Ê
AssignVariableOp_60AssignVariableOpBassignvariableop_60_adam_sequential_1_batch_normalization_1_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61¿
AssignVariableOp_61AssignVariableOp7assignvariableop_61_adam_sequential_3_conv2d_3_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62½
AssignVariableOp_62AssignVariableOp5assignvariableop_62_adam_sequential_3_conv2d_3_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63Ë
AssignVariableOp_63AssignVariableOpCassignvariableop_63_adam_sequential_3_batch_normalization_3_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64Ê
AssignVariableOp_64AssignVariableOpBassignvariableop_64_adam_sequential_3_batch_normalization_3_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65¿
AssignVariableOp_65AssignVariableOp7assignvariableop_65_adam_sequential_5_conv2d_5_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66½
AssignVariableOp_66AssignVariableOp5assignvariableop_66_adam_sequential_5_conv2d_5_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67Ë
AssignVariableOp_67AssignVariableOpCassignvariableop_67_adam_sequential_5_batch_normalization_5_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68Ê
AssignVariableOp_68AssignVariableOpBassignvariableop_68_adam_sequential_5_batch_normalization_5_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69¼
AssignVariableOp_69AssignVariableOp4assignvariableop_69_adam_sequential_6_dense_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70º
AssignVariableOp_70AssignVariableOp2assignvariableop_70_adam_sequential_6_dense_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_709
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpø
Identity_71Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_71ë
Identity_72IdentityIdentity_71:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_72"#
identity_72Identity_72:output:0*³
_input_shapes¡
: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¢
¨
5__inference_batch_normalization_5_layer_call_fn_27818

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_252092
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

«
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27341

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ú
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ü
¨
5__inference_batch_normalization_5_layer_call_fn_27895

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_253562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ@@::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
×

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27529

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ :::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
È
­
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_24855

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

{
&__inference_conv2d_layer_call_fn_27234

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_242192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
©
A__inference_conv2d_layer_call_and_return_conditional_losses_27225

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2	
BiasAddn
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
-
ñ
E__inference_sequential_layer_call_and_return_conditional_losses_26718

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource
identity¢"batch_normalization/AssignNewValue¢$batch_normalization/AssignNewValue_1ª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpº
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¦
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d/BiasAddm
activation_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_1/mul/x
activation_1/mulMulactivation_1/mul/x:output:0conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/mulo
activation_1/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
activation_1/Sqrt/xm
activation_1/SqrtSqrtactivation_1/Sqrt/x:output:0*
T0*
_output_shapes
: 2
activation_1/Sqrtm
activation_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
activation_1/Pow/y
activation_1/PowPowconv2d/BiasAdd:output:0activation_1/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/Powq
activation_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2
activation_1/mul_1/x 
activation_1/mul_1Mulactivation_1/mul_1/x:output:0activation_1/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/mul_1
activation_1/addAddV2conv2d/BiasAdd:output:0activation_1/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/add
activation_1/mul_2Mulactivation_1/Sqrt:y:0activation_1/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/mul_2
activation_1/TanhTanhactivation_1/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/Tanhq
activation_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/add_1/x£
activation_1/add_1AddV2activation_1/add_1/x:output:0activation_1/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/add_1
activation_1/mul_3Mulactivation_1/mul:z:0activation_1/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/mul_3°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1â
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation_1/mul_3:z:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2&
$batch_normalization/FusedBatchNormV3÷
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*O
_classE
CAloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*Q
_classG
ECloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1Ò
IdentityIdentity(batch_normalization/FusedBatchNormV3:y:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
í
G__inference_sequential_1_layer_call_and_return_conditional_losses_24706
conv2d_1_input
conv2d_1_24689
conv2d_1_24691
batch_normalization_1_24695
batch_normalization_1_24697
batch_normalization_1_24699
batch_normalization_1_24701
identity¢-batch_normalization_1/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¦
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_24689conv2d_1_24691*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_245692"
 conv2d_1/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_246032
activation_2/PartitionedCall¼
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0batch_normalization_1_24695batch_normalization_1_24697batch_normalization_1_24699batch_normalization_1_24701*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_246482/
-batch_normalization_1/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_245492
max_pooling2d/PartitionedCall×
IdentityIdentity&max_pooling2d/PartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ ::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_1_input
ï
p
F__inference_concatenate_layer_call_and_return_conditional_losses_25690

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs:YU
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Â
å
G__inference_sequential_3_layer_call_and_return_conditional_losses_25083

inputs
conv2d_3_25066
conv2d_3_25068
batch_normalization_3_25072
batch_normalization_3_25074
batch_normalization_3_25076
batch_normalization_3_25078
identity¢-batch_normalization_3/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_25066conv2d_3_25068*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_249232"
 conv2d_3/StatefulPartitionedCall
activation_4/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_249572
activation_4/PartitionedCallº
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0batch_normalization_3_25072batch_normalization_3_25074batch_normalization_3_25076batch_normalization_3_25078*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_249842/
-batch_normalization_3/StatefulPartitionedCall£
max_pooling2d_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_249032!
max_pooling2d_1/PartitionedCall×
IdentityIdentity(max_pooling2d_1/PartitionedCall:output:0.^batch_normalization_3/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
§
C
'__inference_flatten_layer_call_fn_27906

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_254992
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ú
¦
3__inference_batch_normalization_layer_call_fn_27372

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall£
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_242802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¢
Å
,__inference_sequential_3_layer_call_fn_25135
conv2d_3_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_251202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_3_input
Ô
­
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27787

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
Ó
E__inference_sequential_layer_call_and_return_conditional_losses_24412

inputs
conv2d_24396
conv2d_24398
batch_normalization_24402
batch_normalization_24404
batch_normalization_24406
batch_normalization_24408
identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_24396conv2d_24398*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_242192 
conv2d/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_242532
activation_1/PartitionedCall®
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_24402batch_normalization_24404batch_normalization_24406batch_normalization_24408*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_242982-
+batch_normalization/StatefulPartitionedCallá
IdentityIdentity4batch_normalization/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
å
G__inference_sequential_5_layer_call_and_return_conditional_losses_25474

inputs
conv2d_5_25457
conv2d_5_25459
batch_normalization_5_25463
batch_normalization_5_25465
batch_normalization_5_25467
batch_normalization_5_25469
identity¢-batch_normalization_5/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_25457conv2d_5_25459*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_252772"
 conv2d_5/StatefulPartitionedCall
activation_6/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_253112
activation_6/PartitionedCall»
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0batch_normalization_5_25463batch_normalization_5_25465batch_normalization_5_25467batch_normalization_5_25469*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_253562/
-batch_normalization_5/StatefulPartitionedCall¤
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_252572!
max_pooling2d_2/PartitionedCallØ
IdentityIdentity(max_pooling2d_2/PartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ï
t
H__inference_concatenate_1_layer_call_and_return_conditional_losses_27038
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ@@@:ÿÿÿÿÿÿÿÿÿ@@@:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
"
_user_specified_name
inputs/1

}
(__inference_conv2d_3_layer_call_fn_27574

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_249232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
°
c
G__inference_activation_4_layer_call_and_return_conditional_losses_27592

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xe
mulMulmul/x:output:0inputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mulU
Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
Sqrt/xF
SqrtSqrtSqrt/x:output:0*
T0*
_output_shapes
: 2
SqrtS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
Pow/ye
PowPowinputsPow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
PowW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2	
mul_1/xl
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_1b
addAddV2inputs	mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
addd
mul_2MulSqrt:y:0add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_2[
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
TanhW
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
add_1/xo
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_1e
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_3g
IdentityIdentity	mul_3:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ðô
É
G__inference_functional_1_layer_call_and_return_conditional_losses_26403

inputs4
0sequential_conv2d_conv2d_readvariableop_resource5
1sequential_conv2d_biasadd_readvariableop_resource:
6sequential_batch_normalization_readvariableop_resource<
8sequential_batch_normalization_readvariableop_1_resourceK
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceM
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource8
4sequential_1_conv2d_1_conv2d_readvariableop_resource9
5sequential_1_conv2d_1_biasadd_readvariableop_resource>
:sequential_1_batch_normalization_1_readvariableop_resource@
<sequential_1_batch_normalization_1_readvariableop_1_resourceO
Ksequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource8
4sequential_3_conv2d_3_conv2d_readvariableop_resource9
5sequential_3_conv2d_3_biasadd_readvariableop_resource>
:sequential_3_batch_normalization_3_readvariableop_resource@
<sequential_3_batch_normalization_3_readvariableop_1_resourceO
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceQ
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource8
4sequential_5_conv2d_5_conv2d_readvariableop_resource9
5sequential_5_conv2d_5_biasadd_readvariableop_resource>
:sequential_5_batch_normalization_5_readvariableop_resource@
<sequential_5_batch_normalization_5_readvariableop_1_resourceO
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceQ
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource5
1sequential_6_dense_matmul_readvariableop_resource6
2sequential_6_dense_biasadd_readvariableop_resource
identity¢-sequential/batch_normalization/AssignNewValue¢/sequential/batch_normalization/AssignNewValue_1¢1sequential_1/batch_normalization_1/AssignNewValue¢3sequential_1/batch_normalization_1/AssignNewValue_1¢1sequential_3/batch_normalization_3/AssignNewValue¢3sequential_3/batch_normalization_3/AssignNewValue_1¢1sequential_5/batch_normalization_5/AssignNewValue¢3sequential_5/batch_normalization_5/AssignNewValue_1Ë
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOpÛ
sequential/conv2d/Conv2DConv2Dinputs/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
sequential/conv2d/Conv2DÂ
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOpÒ
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/conv2d/BiasAdd
sequential/activation_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sequential/activation_1/mul/xÉ
sequential/activation_1/mulMul&sequential/activation_1/mul/x:output:0"sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/mul
sequential/activation_1/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2 
sequential/activation_1/Sqrt/x
sequential/activation_1/SqrtSqrt'sequential/activation_1/Sqrt/x:output:0*
T0*
_output_shapes
: 2
sequential/activation_1/Sqrt
sequential/activation_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
sequential/activation_1/Pow/yÉ
sequential/activation_1/PowPow"sequential/conv2d/BiasAdd:output:0&sequential/activation_1/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/Pow
sequential/activation_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2!
sequential/activation_1/mul_1/xÌ
sequential/activation_1/mul_1Mul(sequential/activation_1/mul_1/x:output:0sequential/activation_1/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/mul_1Æ
sequential/activation_1/addAddV2"sequential/conv2d/BiasAdd:output:0!sequential/activation_1/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/addÄ
sequential/activation_1/mul_2Mul sequential/activation_1/Sqrt:y:0sequential/activation_1/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/mul_2£
sequential/activation_1/TanhTanh!sequential/activation_1/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/Tanh
sequential/activation_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2!
sequential/activation_1/add_1/xÏ
sequential/activation_1/add_1AddV2(sequential/activation_1/add_1/x:output:0 sequential/activation_1/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/add_1Å
sequential/activation_1/mul_3Mulsequential/activation_1/mul:z:0!sequential/activation_1/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential/activation_1/mul_3Ñ
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential/batch_normalization/ReadVariableOp×
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype021
/sequential/batch_normalization/ReadVariableOp_1
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1¯
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3!sequential/activation_1/mul_3:z:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<21
/sequential/batch_normalization/FusedBatchNormV3¹
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*Z
_classP
NLloc:@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02/
-sequential/batch_normalization/AssignNewValueÇ
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*\
_classR
PNloc:@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype021
/sequential/batch_normalization/AssignNewValue_1á
max_pooling2d_3/MaxPoolMaxPool3sequential/batch_normalization/FusedBatchNormV3:y:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_3/MaxPool×
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02-
+sequential_1/conv2d_1/Conv2D/ReadVariableOp
sequential_1/conv2d_1/Conv2DConv2D3sequential/batch_normalization/FusedBatchNormV3:y:03sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
sequential_1/conv2d_1/Conv2DÎ
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpâ
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_1/conv2d_1/BiasAdd
sequential_1/activation_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential_1/activation_2/mul/xÓ
sequential_1/activation_2/mulMul(sequential_1/activation_2/mul/x:output:0&sequential_1/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_1/activation_2/mul
 sequential_1/activation_2/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2"
 sequential_1/activation_2/Sqrt/x
sequential_1/activation_2/SqrtSqrt)sequential_1/activation_2/Sqrt/x:output:0*
T0*
_output_shapes
: 2 
sequential_1/activation_2/Sqrt
sequential_1/activation_2/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2!
sequential_1/activation_2/Pow/yÓ
sequential_1/activation_2/PowPow&sequential_1/conv2d_1/BiasAdd:output:0(sequential_1/activation_2/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_1/activation_2/Pow
!sequential_1/activation_2/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2#
!sequential_1/activation_2/mul_1/xÔ
sequential_1/activation_2/mul_1Mul*sequential_1/activation_2/mul_1/x:output:0!sequential_1/activation_2/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential_1/activation_2/mul_1Ð
sequential_1/activation_2/addAddV2&sequential_1/conv2d_1/BiasAdd:output:0#sequential_1/activation_2/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
sequential_1/activation_2/addÌ
sequential_1/activation_2/mul_2Mul"sequential_1/activation_2/Sqrt:y:0!sequential_1/activation_2/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential_1/activation_2/mul_2©
sequential_1/activation_2/TanhTanh#sequential_1/activation_2/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2 
sequential_1/activation_2/Tanh
!sequential_1/activation_2/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!sequential_1/activation_2/add_1/x×
sequential_1/activation_2/add_1AddV2*sequential_1/activation_2/add_1/x:output:0"sequential_1/activation_2/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential_1/activation_2/add_1Í
sequential_1/activation_2/mul_3Mul!sequential_1/activation_2/mul:z:0#sequential_1/activation_2/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2!
sequential_1/activation_2/mul_3Ý
1sequential_1/batch_normalization_1/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype023
1sequential_1/batch_normalization_1/ReadVariableOpã
3sequential_1/batch_normalization_1/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype025
3sequential_1/batch_normalization_1/ReadVariableOp_1
Bsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
Bsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp
Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
Dsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1É
3sequential_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#sequential_1/activation_2/mul_3:z:09sequential_1/batch_normalization_1/ReadVariableOp:value:0;sequential_1/batch_normalization_1/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<25
3sequential_1/batch_normalization_1/FusedBatchNormV3Ñ
1sequential_1/batch_normalization_1/AssignNewValueAssignVariableOpKsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource@sequential_1/batch_normalization_1/FusedBatchNormV3:batch_mean:0C^sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1sequential_1/batch_normalization_1/AssignNewValueß
3sequential_1/batch_normalization_1/AssignNewValue_1AssignVariableOpMsequential_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceDsequential_1/batch_normalization_1/FusedBatchNormV3:batch_variance:0E^sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@sequential_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3sequential_1/batch_normalization_1/AssignNewValue_1û
"sequential_1/max_pooling2d/MaxPoolMaxPool7sequential_1/batch_normalization_1/FusedBatchNormV3:y:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2$
"sequential_1/max_pooling2d/MaxPoolt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axisê
concatenate/concatConcatV2 max_pooling2d_3/MaxPool:output:0+sequential_1/max_pooling2d/MaxPool:output:0 concatenate/concat/axis:output:0*
N*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
concatenate/concatÇ
max_pooling2d_4/MaxPoolMaxPoolconcatenate/concat:output:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_4/MaxPool×
+sequential_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4sequential_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+sequential_3/conv2d_3/Conv2D/ReadVariableOpü
sequential_3/conv2d_3/Conv2DConv2Dconcatenate/concat:output:03sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
sequential_3/conv2d_3/Conv2DÎ
,sequential_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5sequential_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_3/conv2d_3/BiasAdd/ReadVariableOpâ
sequential_3/conv2d_3/BiasAddBiasAdd%sequential_3/conv2d_3/Conv2D:output:04sequential_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_3/conv2d_3/BiasAdd
sequential_3/activation_4/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential_3/activation_4/mul/xÓ
sequential_3/activation_4/mulMul(sequential_3/activation_4/mul/x:output:0&sequential_3/conv2d_3/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_3/activation_4/mul
 sequential_3/activation_4/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2"
 sequential_3/activation_4/Sqrt/x
sequential_3/activation_4/SqrtSqrt)sequential_3/activation_4/Sqrt/x:output:0*
T0*
_output_shapes
: 2 
sequential_3/activation_4/Sqrt
sequential_3/activation_4/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2!
sequential_3/activation_4/Pow/yÓ
sequential_3/activation_4/PowPow&sequential_3/conv2d_3/BiasAdd:output:0(sequential_3/activation_4/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_3/activation_4/Pow
!sequential_3/activation_4/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2#
!sequential_3/activation_4/mul_1/xÔ
sequential_3/activation_4/mul_1Mul*sequential_3/activation_4/mul_1/x:output:0!sequential_3/activation_4/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_3/activation_4/mul_1Ð
sequential_3/activation_4/addAddV2&sequential_3/conv2d_3/BiasAdd:output:0#sequential_3/activation_4/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_3/activation_4/addÌ
sequential_3/activation_4/mul_2Mul"sequential_3/activation_4/Sqrt:y:0!sequential_3/activation_4/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_3/activation_4/mul_2©
sequential_3/activation_4/TanhTanh#sequential_3/activation_4/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
sequential_3/activation_4/Tanh
!sequential_3/activation_4/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!sequential_3/activation_4/add_1/x×
sequential_3/activation_4/add_1AddV2*sequential_3/activation_4/add_1/x:output:0"sequential_3/activation_4/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_3/activation_4/add_1Í
sequential_3/activation_4/mul_3Mul!sequential_3/activation_4/mul:z:0#sequential_3/activation_4/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_3/activation_4/mul_3Ý
1sequential_3/batch_normalization_3/ReadVariableOpReadVariableOp:sequential_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential_3/batch_normalization_3/ReadVariableOpã
3sequential_3/batch_normalization_3/ReadVariableOp_1ReadVariableOp<sequential_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3sequential_3/batch_normalization_3/ReadVariableOp_1
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1É
3sequential_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3#sequential_3/activation_4/mul_3:z:09sequential_3/batch_normalization_3/ReadVariableOp:value:0;sequential_3/batch_normalization_3/ReadVariableOp_1:value:0Jsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<25
3sequential_3/batch_normalization_3/FusedBatchNormV3Ñ
1sequential_3/batch_normalization_3/AssignNewValueAssignVariableOpKsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource@sequential_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0C^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1sequential_3/batch_normalization_3/AssignNewValueß
3sequential_3/batch_normalization_3/AssignNewValue_1AssignVariableOpMsequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceDsequential_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0E^sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@sequential_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3sequential_3/batch_normalization_3/AssignNewValue_1ý
$sequential_3/max_pooling2d_1/MaxPoolMaxPool7sequential_3/batch_normalization_3/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*
ksize
*
paddingVALID*
strides
2&
$sequential_3/max_pooling2d_1/MaxPoolx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axisñ
concatenate_1/concatConcatV2 max_pooling2d_4/MaxPool:output:0-sequential_3/max_pooling2d_1/MaxPool:output:0"concatenate_1/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
concatenate_1/concatÊ
max_pooling2d_5/MaxPoolMaxPoolconcatenate_1/concat:output:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_5/MaxPoolÙ
+sequential_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4sequential_5_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+sequential_5/conv2d_5/Conv2D/ReadVariableOpý
sequential_5/conv2d_5/Conv2DConv2Dconcatenate_1/concat:output:03sequential_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
2
sequential_5/conv2d_5/Conv2DÏ
,sequential_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,sequential_5/conv2d_5/BiasAdd/ReadVariableOpá
sequential_5/conv2d_5/BiasAddBiasAdd%sequential_5/conv2d_5/Conv2D:output:04sequential_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_5/conv2d_5/BiasAdd
sequential_5/activation_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential_5/activation_6/mul/xÒ
sequential_5/activation_6/mulMul(sequential_5/activation_6/mul/x:output:0&sequential_5/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_5/activation_6/mul
 sequential_5/activation_6/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2"
 sequential_5/activation_6/Sqrt/x
sequential_5/activation_6/SqrtSqrt)sequential_5/activation_6/Sqrt/x:output:0*
T0*
_output_shapes
: 2 
sequential_5/activation_6/Sqrt
sequential_5/activation_6/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2!
sequential_5/activation_6/Pow/yÒ
sequential_5/activation_6/PowPow&sequential_5/conv2d_5/BiasAdd:output:0(sequential_5/activation_6/Pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_5/activation_6/Pow
!sequential_5/activation_6/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2#
!sequential_5/activation_6/mul_1/xÓ
sequential_5/activation_6/mul_1Mul*sequential_5/activation_6/mul_1/x:output:0!sequential_5/activation_6/Pow:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_5/activation_6/mul_1Ï
sequential_5/activation_6/addAddV2&sequential_5/conv2d_5/BiasAdd:output:0#sequential_5/activation_6/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_5/activation_6/addË
sequential_5/activation_6/mul_2Mul"sequential_5/activation_6/Sqrt:y:0!sequential_5/activation_6/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_5/activation_6/mul_2¨
sequential_5/activation_6/TanhTanh#sequential_5/activation_6/mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2 
sequential_5/activation_6/Tanh
!sequential_5/activation_6/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!sequential_5/activation_6/add_1/xÖ
sequential_5/activation_6/add_1AddV2*sequential_5/activation_6/add_1/x:output:0"sequential_5/activation_6/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_5/activation_6/add_1Ì
sequential_5/activation_6/mul_3Mul!sequential_5/activation_6/mul:z:0#sequential_5/activation_6/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_5/activation_6/mul_3Þ
1sequential_5/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_5_batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype023
1sequential_5/batch_normalization_5/ReadVariableOpä
3sequential_5/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_5_batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3sequential_5/batch_normalization_5/ReadVariableOp_1
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ì
3sequential_5/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3#sequential_5/activation_6/mul_3:z:09sequential_5/batch_normalization_5/ReadVariableOp:value:0;sequential_5/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
exponential_avg_factor%
×#<25
3sequential_5/batch_normalization_5/FusedBatchNormV3Ñ
1sequential_5/batch_normalization_5/AssignNewValueAssignVariableOpKsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resource@sequential_5/batch_normalization_5/FusedBatchNormV3:batch_mean:0C^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1sequential_5/batch_normalization_5/AssignNewValueß
3sequential_5/batch_normalization_5/AssignNewValue_1AssignVariableOpMsequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceDsequential_5/batch_normalization_5/FusedBatchNormV3:batch_variance:0E^sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@sequential_5/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3sequential_5/batch_normalization_5/AssignNewValue_1þ
$sequential_5/max_pooling2d_2/MaxPoolMaxPool7sequential_5/batch_normalization_5/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
2&
$sequential_5/max_pooling2d_2/MaxPoolx
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axisñ
concatenate_2/concatConcatV2 max_pooling2d_5/MaxPool:output:0-sequential_5/max_pooling2d_2/MaxPool:output:0"concatenate_2/concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
concatenate_2/concat
sequential_6/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
sequential_6/flatten/Const¿
sequential_6/flatten/ReshapeReshapeconcatenate_2/concat:output:0#sequential_6/flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/flatten/ReshapeÈ
(sequential_6/dense/MatMul/ReadVariableOpReadVariableOp1sequential_6_dense_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02*
(sequential_6/dense/MatMul/ReadVariableOpË
sequential_6/dense/MatMulMatMul%sequential_6/flatten/Reshape:output:00sequential_6/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/dense/MatMulÅ
)sequential_6/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_6_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential_6/dense/BiasAdd/ReadVariableOpÍ
sequential_6/dense/BiasAddBiasAdd#sequential_6/dense/MatMul:product:01sequential_6/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/dense/BiasAdd
sequential_6/dense/SoftmaxSoftmax#sequential_6/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/dense/Softmax
IdentityIdentity$sequential_6/dense/Softmax:softmax:0.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_12^sequential_1/batch_normalization_1/AssignNewValue4^sequential_1/batch_normalization_1/AssignNewValue_12^sequential_3/batch_normalization_3/AssignNewValue4^sequential_3/batch_normalization_3/AssignNewValue_12^sequential_5/batch_normalization_5/AssignNewValue4^sequential_5/batch_normalization_5/AssignNewValue_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12f
1sequential_1/batch_normalization_1/AssignNewValue1sequential_1/batch_normalization_1/AssignNewValue2j
3sequential_1/batch_normalization_1/AssignNewValue_13sequential_1/batch_normalization_1/AssignNewValue_12f
1sequential_3/batch_normalization_3/AssignNewValue1sequential_3/batch_normalization_3/AssignNewValue2j
3sequential_3/batch_normalization_3/AssignNewValue_13sequential_3/batch_normalization_3/AssignNewValue_12f
1sequential_5/batch_normalization_5/AssignNewValue1sequential_5/batch_normalization_5/AssignNewValue2j
3sequential_5/batch_normalization_5/AssignNewValue_13sequential_5/batch_normalization_5/AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

½
,__inference_sequential_1_layer_call_fn_26904

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_247662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ú
W
+__inference_concatenate_layer_call_fn_26917
inputs_0
inputs_1
identityÞ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_256902
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*M
_input_shapes<
::ÿÿÿÿÿÿÿÿÿ :ÿÿÿÿÿÿÿÿÿ :[ W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"
_user_specified_name
inputs/1

­
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27511

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ú
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¨
5__inference_batch_normalization_1_layer_call_fn_27478

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_245012
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
³
¨
@__inference_dense_layer_call_and_return_conditional_losses_25518

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
í
G__inference_sequential_1_layer_call_and_return_conditional_losses_24686
conv2d_1_input
conv2d_1_24580
conv2d_1_24582
batch_normalization_1_24675
batch_normalization_1_24677
batch_normalization_1_24679
batch_normalization_1_24681
identity¢-batch_normalization_1/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¦
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_24580conv2d_1_24582*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_245692"
 conv2d_1/StatefulPartitionedCall
activation_2/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_246032
activation_2/PartitionedCallº
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0batch_normalization_1_24675batch_normalization_1_24677batch_normalization_1_24679batch_normalization_1_24681*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_246302/
-batch_normalization_1/StatefulPartitionedCall
max_pooling2d/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_245492
max_pooling2d/PartitionedCall×
IdentityIdentity&max_pooling2d/PartitionedCall:output:0.^batch_normalization_1/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ ::::::2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
(
_user_specified_nameconv2d_1_input
´

¨
G__inference_sequential_6_layer_call_and_return_conditional_losses_25577

inputs
dense_25571
dense_25573
identity¢dense/StatefulPartitionedCallÕ
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_254992
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_25571dense_25573*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_255182
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ  ::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
 
Å
,__inference_sequential_3_layer_call_fn_25098
conv2d_3_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¼
StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_3_layer_call_and_return_conditional_losses_250832
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_3_input
ò
Ó
E__inference_sequential_layer_call_and_return_conditional_losses_24376

inputs
conv2d_24360
conv2d_24362
batch_normalization_24366
batch_normalization_24368
batch_normalization_24370
batch_normalization_24372
identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_24360conv2d_24362*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_242192 
conv2d/StatefulPartitionedCall
activation_1/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_242532
activation_1/PartitionedCall¬
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_24366batch_normalization_24368batch_normalization_24370batch_normalization_24372*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_242802-
+batch_normalization/StatefulPartitionedCallá
IdentityIdentity4batch_normalization/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
G__inference_activation_2_layer_call_and_return_conditional_losses_24603

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xe
mulMulmul/x:output:0inputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulU
Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
Sqrt/xF
SqrtSqrtSqrt/x:output:0*
T0*
_output_shapes
: 2
SqrtS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
Pow/ye
PowPowinputsPow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
PowW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2	
mul_1/xl
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1b
addAddV2inputs	mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
addd
mul_2MulSqrt:y:0add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2[
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
TanhW
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
add_1/xo
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1e
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3g
IdentityIdentity	mul_3:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¦
3__inference_batch_normalization_layer_call_fn_27321

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallµ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_241942
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¿
H
,__inference_activation_6_layer_call_fn_27767

inputs
identityÑ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_253112
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Û
z
%__inference_dense_layer_call_fn_27926

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_255182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
í
G__inference_sequential_5_layer_call_and_return_conditional_losses_25394
conv2d_5_input
conv2d_5_25288
conv2d_5_25290
batch_normalization_5_25383
batch_normalization_5_25385
batch_normalization_5_25387
batch_normalization_5_25389
identity¢-batch_normalization_5/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall¥
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_25288conv2d_5_25290*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_252772"
 conv2d_5/StatefulPartitionedCall
activation_6/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_253112
activation_6/PartitionedCall¹
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0batch_normalization_5_25383batch_normalization_5_25385batch_normalization_5_25387batch_normalization_5_25389*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_253382/
-batch_normalization_5/StatefulPartitionedCall¤
max_pooling2d_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_252572!
max_pooling2d_2/PartitionedCallØ
IdentityIdentity(max_pooling2d_2/PartitionedCall:output:0.^batch_normalization_5/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
(
_user_specified_nameconv2d_5_input
ç&
³
G__inference_sequential_5_layer_call_and_return_conditional_losses_27124

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity²
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOp¿
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
2
conv2d_5/Conv2D¨
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp­
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
conv2d_5/BiasAddm
activation_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_6/mul/x
activation_6/mulMulactivation_6/mul/x:output:0conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/mulo
activation_6/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
activation_6/Sqrt/xm
activation_6/SqrtSqrtactivation_6/Sqrt/x:output:0*
T0*
_output_shapes
: 2
activation_6/Sqrtm
activation_6/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
activation_6/Pow/y
activation_6/PowPowconv2d_5/BiasAdd:output:0activation_6/Pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/Powq
activation_6/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2
activation_6/mul_1/x
activation_6/mul_1Mulactivation_6/mul_1/x:output:0activation_6/Pow:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/mul_1
activation_6/addAddV2conv2d_5/BiasAdd:output:0activation_6/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/add
activation_6/mul_2Mulactivation_6/Sqrt:y:0activation_6/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/mul_2
activation_6/TanhTanhactivation_6/mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/Tanhq
activation_6/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_6/add_1/x¢
activation_6/add_1AddV2activation_6/add_1/x:output:0activation_6/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/add_1
activation_6/mul_3Mulactivation_6/mul:z:0activation_6/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/mul_3·
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_5/ReadVariableOp½
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1ê
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ã
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_6/mul_3:z:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3×
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_5/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPool}
IdentityIdentity max_pooling2d_2/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@:::::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
à
¨
5__inference_batch_normalization_3_layer_call_fn_27725

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_250022
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Û

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_25356

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ@@:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs


P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27635

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 
¨
5__inference_batch_normalization_3_layer_call_fn_27661

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_248862
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
®
«
C__inference_conv2d_3_layer_call_and_return_conditional_losses_27565

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¥
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddn
IdentityIdentityBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:ÿÿÿÿÿÿÿÿÿ@:::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
å&
³
G__inference_sequential_1_layer_call_and_return_conditional_losses_26870

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpÀ
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_1/BiasAdd/ReadVariableOp®
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d_1/BiasAddm
activation_2/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_2/mul/x
activation_2/mulMulactivation_2/mul/x:output:0conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/mulo
activation_2/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
activation_2/Sqrt/xm
activation_2/SqrtSqrtactivation_2/Sqrt/x:output:0*
T0*
_output_shapes
: 2
activation_2/Sqrtm
activation_2/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
activation_2/Pow/y
activation_2/PowPowconv2d_1/BiasAdd:output:0activation_2/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/Powq
activation_2/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2
activation_2/mul_1/x 
activation_2/mul_1Mulactivation_2/mul_1/x:output:0activation_2/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/mul_1
activation_2/addAddV2conv2d_1/BiasAdd:output:0activation_2/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/add
activation_2/mul_2Mulactivation_2/Sqrt:y:0activation_2/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/mul_2
activation_2/TanhTanhactivation_2/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/Tanhq
activation_2/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_2/add_1/x£
activation_2/add_1AddV2activation_2/add_1/x:output:0activation_2/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/add_1
activation_2/mul_3Mulactivation_2/mul:z:0activation_2/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_2/mul_3¶
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp¼
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1é
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1à
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_2/mul_3:z:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3Ô
max_pooling2d/MaxPoolMaxPool*batch_normalization_1/FusedBatchNormV3:y:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool|
IdentityIdentitymax_pooling2d/MaxPool:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ :::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_25141

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
r
H__inference_concatenate_2_layer_call_and_return_conditional_losses_25818

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs:XT
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs

½
,__inference_sequential_1_layer_call_fn_26887

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_247292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ ::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Û

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27869

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ@@:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
È
­
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_24501

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

¦
3__inference_batch_normalization_layer_call_fn_27308

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_241632
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ø
Y
-__inference_concatenate_2_layer_call_fn_27171
inputs_0
inputs_1
identityß
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_2_layer_call_and_return_conditional_losses_258182
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:ÿÿÿÿÿÿÿÿÿ  :ÿÿÿÿÿÿÿÿÿ  :Z V
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
"
_user_specified_name
inputs/1

}
(__inference_conv2d_5_layer_call_fn_27744

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_conv2d_5_layer_call_and_return_conditional_losses_252772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ@@::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs

­
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_25338

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ@@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Ç$
¥
E__inference_sequential_layer_call_and_return_conditional_losses_26756

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource
identityª
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOpº
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
paddingSAME*
strides
2
conv2d/Conv2D¡
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp¦
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
conv2d/BiasAddm
activation_1/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_1/mul/x
activation_1/mulMulactivation_1/mul/x:output:0conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/mulo
activation_1/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
activation_1/Sqrt/xm
activation_1/SqrtSqrtactivation_1/Sqrt/x:output:0*
T0*
_output_shapes
: 2
activation_1/Sqrtm
activation_1/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
activation_1/Pow/y
activation_1/PowPowconv2d/BiasAdd:output:0activation_1/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/Powq
activation_1/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2
activation_1/mul_1/x 
activation_1/mul_1Mulactivation_1/mul_1/x:output:0activation_1/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/mul_1
activation_1/addAddV2conv2d/BiasAdd:output:0activation_1/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/add
activation_1/mul_2Mulactivation_1/Sqrt:y:0activation_1/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/mul_2
activation_1/TanhTanhactivation_1/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/Tanhq
activation_1/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_1/add_1/x£
activation_1/add_1AddV2activation_1/add_1/x:output:0activation_1/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/add_1
activation_1/mul_3Mulactivation_1/mul:z:0activation_1/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
activation_1/mul_3°
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp¶
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1ã
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOpé
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ô
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation_1/mul_3:z:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3
IdentityIdentity(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ:::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
c
G__inference_activation_2_layer_call_and_return_conditional_losses_27422

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xe
mulMulmul/x:output:0inputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mulU
Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
Sqrt/xF
SqrtSqrtSqrt/x:output:0*
T0*
_output_shapes
: 2
SqrtS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
Pow/ye
PowPowinputsPow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
PowW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2	
mul_1/xl
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_1b
addAddV2inputs	mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
addd
mul_2MulSqrt:y:0add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_2[
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
TanhW
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
add_1/xo
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
add_1e
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2
mul_3g
IdentityIdentity	mul_3:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬
«
C__inference_conv2d_5_layer_call_and_return_conditional_losses_25277

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2	
BiasAddm
IdentityIdentityBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ@@:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
Õ

N__inference_batch_normalization_layer_call_and_return_conditional_losses_27359

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ :::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

«
N__inference_batch_normalization_layer_call_and_return_conditional_losses_24280

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ú
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¾
^
B__inference_flatten_layer_call_and_return_conditional_losses_27901

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ô
­
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_25209

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
Y
-__inference_concatenate_1_layer_call_fn_27044
inputs_0
inputs_1
identityß
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_257542
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:ÿÿÿÿÿÿÿÿÿ@@@:ÿÿÿÿÿÿÿÿÿ@@@:Y U
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@@
"
_user_specified_name
inputs/1
®
K
/__inference_max_pooling2d_5_layer_call_fn_25147

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_251412
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ

N__inference_batch_normalization_layer_call_and_return_conditional_losses_24298

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ì
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3r
IdentityIdentityFusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ :::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ð/

G__inference_sequential_5_layer_call_and_return_conditional_losses_27085

inputs+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_5/AssignNewValue¢&batch_normalization_5/AssignNewValue_1²
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOp¿
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
2
conv2d_5/Conv2D¨
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp­
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
conv2d_5/BiasAddm
activation_6/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_6/mul/x
activation_6/mulMulactivation_6/mul/x:output:0conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/mulo
activation_6/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
activation_6/Sqrt/xm
activation_6/SqrtSqrtactivation_6/Sqrt/x:output:0*
T0*
_output_shapes
: 2
activation_6/Sqrtm
activation_6/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
activation_6/Pow/y
activation_6/PowPowconv2d_5/BiasAdd:output:0activation_6/Pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/Powq
activation_6/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2
activation_6/mul_1/x
activation_6/mul_1Mulactivation_6/mul_1/x:output:0activation_6/Pow:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/mul_1
activation_6/addAddV2conv2d_5/BiasAdd:output:0activation_6/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/add
activation_6/mul_2Mulactivation_6/Sqrt:y:0activation_6/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/mul_2
activation_6/TanhTanhactivation_6/mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/Tanhq
activation_6/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_6/add_1/x¢
activation_6/add_1AddV2activation_6/add_1/x:output:0activation_6/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/add_1
activation_6/mul_3Mulactivation_6/mul:z:0activation_6/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_6/mul_3·
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_5/ReadVariableOp½
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_5/ReadVariableOp_1ê
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ñ
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_6/mul_3:z:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_5/FusedBatchNormV3
$batch_normalization_5/AssignNewValueAssignVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValue
&batch_normalization_5/AssignNewValue_1AssignVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1×
max_pooling2d_2/MaxPoolMaxPool*batch_normalization_5/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *
ksize
*
paddingVALID*
strides
2
max_pooling2d_2/MaxPoolÍ
IdentityIdentity max_pooling2d_2/MaxPool:output:0%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::2L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
¾
^
B__inference_flatten_layer_call_and_return_conditional_losses_25499

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ  :X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  
 
_user_specified_nameinputs
Ã
H
,__inference_activation_4_layer_call_fn_27597

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_249572
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_24787

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
c
G__inference_activation_6_layer_call_and_return_conditional_losses_25311

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xd
mulMulmul/x:output:0inputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
mulU
Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
Sqrt/xF
SqrtSqrtSqrt/x:output:0*
T0*
_output_shapes
: 2
SqrtS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
Pow/yd
PowPowinputsPow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
PowW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2	
mul_1/xk
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
mul_1a
addAddV2inputs	mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
addc
mul_2MulSqrt:y:0add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
mul_2Z
TanhTanh	mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
TanhW
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
add_1/xn
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
add_1d
mul_3Mulmul:z:0	add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
mul_3f
IdentityIdentity	mul_3:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@@:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
®
K
/__inference_max_pooling2d_3_layer_call_fn_24439

inputs
identityî
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_244332
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
¨
5__inference_batch_normalization_1_layer_call_fn_27555

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_246482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs

ø
,__inference_functional_1_layer_call_fn_26678

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_261122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

½
,__inference_sequential_5_layer_call_fn_27158

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_5_layer_call_and_return_conditional_losses_254742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ  2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
°
c
G__inference_activation_4_layer_call_and_return_conditional_losses_24957

inputs
identityS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xe
mulMulmul/x:output:0inputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mulU
Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
Sqrt/xF
SqrtSqrtSqrt/x:output:0*
T0*
_output_shapes
: 2
SqrtS
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
Pow/ye
PowPowinputsPow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
PowW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2	
mul_1/xl
mul_1Mulmul_1/x:output:0Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_1b
addAddV2inputs	mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
addd
mul_2MulSqrt:y:0add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_2[
TanhTanh	mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
TanhW
add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2	
add_1/xo
add_1AddV2add_1/x:output:0Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
add_1e
mul_3Mulmul:z:0	add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
mul_3g
IdentityIdentity	mul_3:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¹
serving_default¥
E
input_1:
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ@
sequential_60
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:»ñ
²
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
_default_save_signature
__call__
+&call_and_return_all_conditional_losses"®
_tf_keras_networkþ­{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["sequential", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_1", "inbound_nodes": [[["sequential", 1, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}], ["sequential_1", 1, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_3", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}], ["sequential_3", 1, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_5", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}], ["sequential_5", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_input"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 29, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_6", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["sequential_6", 1, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["sequential", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_1", "inbound_nodes": [[["sequential", 1, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}], ["sequential_1", 1, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_3", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}], ["sequential_3", 1, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_5", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}], ["sequential_5", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_input"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 29, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_6", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["sequential_6", 1, 0]]}}, "training_config": {"loss": {"class_name": "CategoricalCrossentropy", "config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0.1}}, "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ý"ú
_tf_keras_input_layerÚ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
¸
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ì
_tf_keras_sequential­{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}}}

trainable_variables
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
$
layer_with_weights-0
layer-0
layer-1
 layer_with_weights-1
 layer-2
!layer-3
"trainable_variables
#	variables
$regularization_losses
%	keras_api
__call__
+&call_and_return_all_conditional_losses""
_tf_keras_sequentialë!{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}}
ß
&trainable_variables
'	variables
(regularization_losses
)	keras_api
+&call_and_return_all_conditional_losses
__call__"Î
_tf_keras_layer´{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 128, 32]}, {"class_name": "TensorShape", "items": [null, 128, 128, 32]}]}

*trainable_variables
+	variables
,regularization_losses
-	keras_api
+&call_and_return_all_conditional_losses
__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
$
.layer_with_weights-0
.layer-0
/layer-1
0layer_with_weights-1
0layer-2
1layer-3
2trainable_variables
3	variables
4regularization_losses
5	keras_api
__call__
+&call_and_return_all_conditional_losses""
_tf_keras_sequentialï!{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}}
ß
6trainable_variables
7	variables
8regularization_losses
9	keras_api
+&call_and_return_all_conditional_losses
__call__"Î
_tf_keras_layer´{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64, 64, 64]}, {"class_name": "TensorShape", "items": [null, 64, 64, 64]}]}

:trainable_variables
;	variables
<regularization_losses
=	keras_api
+&call_and_return_all_conditional_losses
__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
$
>layer_with_weights-0
>layer-0
?layer-1
@layer_with_weights-1
@layer-2
Alayer-3
Btrainable_variables
C	variables
Dregularization_losses
E	keras_api
__call__
+&call_and_return_all_conditional_losses""
_tf_keras_sequentialï!{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}}
á
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
+&call_and_return_all_conditional_losses
__call__"Ð
_tf_keras_layer¶{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 32, 128]}, {"class_name": "TensorShape", "items": [null, 32, 32, 128]}]}
¾
Jlayer-0
Klayer_with_weights-0
Klayer-1
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
__call__
+ &call_and_return_all_conditional_losses"ù
_tf_keras_sequentialÚ{"class_name": "Sequential", "name": "sequential_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_input"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 29, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 256]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_input"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 29, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
»
Piter

Qbeta_1

Rbeta_2
	Sdecay
Tlearning_rateUmäVmåWmæXmçYmèZmé[mê\më]mì^mí_mî`mïamðbmñcmòdmóemôfmõUvöVv÷WvøXvùYvúZvû[vü\vý]vþ^vÿ_v`vavbvcvdvevfv"
	optimizer
¦
U0
V1
W2
X3
Y4
Z5
[6
\7
]8
^9
_10
`11
a12
b13
c14
d15
e16
f17"
trackable_list_wrapper
æ
U0
V1
W2
X3
g4
h5
Y6
Z7
[8
\9
i10
j11
]12
^13
_14
`15
k16
l17
a18
b19
c20
d21
m22
n23
e24
f25"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
ometrics
trainable_variables

players
qlayer_metrics
rnon_trainable_variables
	variables
regularization_losses
slayer_regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
¡serving_default"
signature_map


t_inbound_nodes

Ukernel
Vbias
utrainable_variables
v	variables
wregularization_losses
x	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"Ë
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}}
ò
y_inbound_nodes
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
+¤&call_and_return_all_conditional_losses
¥__call__"Í
_tf_keras_layer³{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}
Ò	
~_inbound_nodes
axis
	Wgamma
Xbeta
gmoving_mean
hmoving_variance
trainable_variables
	variables
regularization_losses
	keras_api
+¦&call_and_return_all_conditional_losses
§__call__"ä
_tf_keras_layerÊ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 32]}}
<
U0
V1
W2
X3"
trackable_list_wrapper
J
U0
V1
W2
X3
g4
h5"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
trainable_variables
layers
layer_metrics
non_trainable_variables
	variables
regularization_losses
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
trainable_variables
layers
layer_metrics
non_trainable_variables
	variables
regularization_losses
 layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


_inbound_nodes

Ykernel
Zbias
trainable_variables
	variables
regularization_losses
	keras_api
+¨&call_and_return_all_conditional_losses
©__call__"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 32]}}
÷
_inbound_nodes
trainable_variables
	variables
regularization_losses
	keras_api
+ª&call_and_return_all_conditional_losses
«__call__"Í
_tf_keras_layer³{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}
Ø	
_inbound_nodes
	axis
	[gamma
\beta
imoving_mean
jmoving_variance
trainable_variables
	variables
regularization_losses
	keras_api
+¬&call_and_return_all_conditional_losses
­__call__"è
_tf_keras_layerÎ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 32]}}

_inbound_nodes
trainable_variables
 	variables
¡regularization_losses
¢	keras_api
+®&call_and_return_all_conditional_losses
¯__call__"ì
_tf_keras_layerÒ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
<
Y0
Z1
[2
\3"
trackable_list_wrapper
J
Y0
Z1
[2
\3
i4
j5"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
£metrics
"trainable_variables
¤layers
¥layer_metrics
¦non_trainable_variables
#	variables
$regularization_losses
 §layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¨metrics
&trainable_variables
©layers
ªlayer_metrics
«non_trainable_variables
'	variables
(regularization_losses
 ¬layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
­metrics
*trainable_variables
®layers
¯layer_metrics
°non_trainable_variables
+	variables
,regularization_losses
 ±layer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


²_inbound_nodes

]kernel
^bias
³trainable_variables
´	variables
µregularization_losses
¶	keras_api
+°&call_and_return_all_conditional_losses
±__call__"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 64]}}
÷
·_inbound_nodes
¸trainable_variables
¹	variables
ºregularization_losses
»	keras_api
+²&call_and_return_all_conditional_losses
³__call__"Í
_tf_keras_layer³{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}
Ø	
¼_inbound_nodes
	½axis
	_gamma
`beta
kmoving_mean
lmoving_variance
¾trainable_variables
¿	variables
Àregularization_losses
Á	keras_api
+´&call_and_return_all_conditional_losses
µ__call__"è
_tf_keras_layerÎ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 64]}}

Â_inbound_nodes
Ãtrainable_variables
Ä	variables
Åregularization_losses
Æ	keras_api
+¶&call_and_return_all_conditional_losses
·__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
<
]0
^1
_2
`3"
trackable_list_wrapper
J
]0
^1
_2
`3
k4
l5"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Çmetrics
2trainable_variables
Èlayers
Élayer_metrics
Ênon_trainable_variables
3	variables
4regularization_losses
 Ëlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ìmetrics
6trainable_variables
Ílayers
Îlayer_metrics
Ïnon_trainable_variables
7	variables
8regularization_losses
 Ðlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ñmetrics
:trainable_variables
Òlayers
Ólayer_metrics
Ônon_trainable_variables
;	variables
<regularization_losses
 Õlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


Ö_inbound_nodes

akernel
bbias
×trainable_variables
Ø	variables
Ùregularization_losses
Ú	keras_api
+¸&call_and_return_all_conditional_losses
¹__call__"Ò
_tf_keras_layer¸{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 128]}}
÷
Û_inbound_nodes
Ütrainable_variables
Ý	variables
Þregularization_losses
ß	keras_api
+º&call_and_return_all_conditional_losses
»__call__"Í
_tf_keras_layer³{"class_name": "Activation", "name": "activation_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}
Ø	
à_inbound_nodes
	áaxis
	cgamma
dbeta
mmoving_mean
nmoving_variance
âtrainable_variables
ã	variables
äregularization_losses
å	keras_api
+¼&call_and_return_all_conditional_losses
½__call__"è
_tf_keras_layerÎ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 128]}}

æ_inbound_nodes
çtrainable_variables
è	variables
éregularization_losses
ê	keras_api
+¾&call_and_return_all_conditional_losses
¿__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
<
a0
b1
c2
d3"
trackable_list_wrapper
J
a0
b1
c2
d3
m4
n5"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ëmetrics
Btrainable_variables
ìlayers
ílayer_metrics
înon_trainable_variables
C	variables
Dregularization_losses
 ïlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ðmetrics
Ftrainable_variables
ñlayers
òlayer_metrics
ónon_trainable_variables
G	variables
Hregularization_losses
 ôlayer_regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ý
õ_inbound_nodes
ötrainable_variables
÷	variables
øregularization_losses
ù	keras_api
+À&call_and_return_all_conditional_losses
Á__call__"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}

ú_inbound_nodes

ekernel
fbias
ûtrainable_variables
ü	variables
ýregularization_losses
þ	keras_api
+Â&call_and_return_all_conditional_losses
Ã__call__"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 29, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 262144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 262144]}}
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ÿmetrics
Ltrainable_variables
layers
layer_metrics
non_trainable_variables
M	variables
Nregularization_losses
 layer_regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
2:0 2sequential/conv2d/kernel
$:" 2sequential/conv2d/bias
2:0 2$sequential/batch_normalization/gamma
1:/ 2#sequential/batch_normalization/beta
6:4  2sequential_1/conv2d_1/kernel
(:& 2sequential_1/conv2d_1/bias
6:4 2(sequential_1/batch_normalization_1/gamma
5:3 2'sequential_1/batch_normalization_1/beta
6:4@@2sequential_3/conv2d_3/kernel
(:&@2sequential_3/conv2d_3/bias
6:4@2(sequential_3/batch_normalization_3/gamma
5:3@2'sequential_3/batch_normalization_3/beta
8:62sequential_5/conv2d_5/kernel
):'2sequential_5/conv2d_5/bias
7:52(sequential_5/batch_normalization_5/gamma
6:42'sequential_5/batch_normalization_5/beta
-:+
2sequential_6/dense/kernel
%:#2sequential_6/dense/bias
::8  (2*sequential/batch_normalization/moving_mean
>:<  (2.sequential/batch_normalization/moving_variance
>:<  (2.sequential_1/batch_normalization_1/moving_mean
B:@  (22sequential_1/batch_normalization_1/moving_variance
>:<@ (2.sequential_3/batch_normalization_3/moving_mean
B:@@ (22sequential_3/batch_normalization_3/moving_variance
?:= (2.sequential_5/batch_normalization_5/moving_mean
C:A (22sequential_5/batch_normalization_5/moving_variance
0
0
1"
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
trackable_list_wrapper
 "
trackable_dict_wrapper
X
g0
h1
i2
j3
k4
l5
m6
n7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
utrainable_variables
layers
layer_metrics
non_trainable_variables
v	variables
wregularization_losses
 layer_regularization_losses
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
ztrainable_variables
layers
layer_metrics
non_trainable_variables
{	variables
|regularization_losses
 layer_regularization_losses
¥__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
<
W0
X1
g2
h3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
metrics
trainable_variables
layers
layer_metrics
non_trainable_variables
	variables
regularization_losses
 layer_regularization_losses
§__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
metrics
trainable_variables
layers
layer_metrics
non_trainable_variables
	variables
regularization_losses
 layer_regularization_losses
©__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
metrics
trainable_variables
layers
layer_metrics
non_trainable_variables
	variables
regularization_losses
 layer_regularization_losses
«__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
<
[0
\1
i2
j3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
metrics
trainable_variables
 layers
¡layer_metrics
¢non_trainable_variables
	variables
regularization_losses
 £layer_regularization_losses
­__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¤metrics
trainable_variables
¥layers
¦layer_metrics
§non_trainable_variables
 	variables
¡regularization_losses
 ¨layer_regularization_losses
¯__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
 2
!3"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
©metrics
³trainable_variables
ªlayers
«layer_metrics
¬non_trainable_variables
´	variables
µregularization_losses
 ­layer_regularization_losses
±__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
®metrics
¸trainable_variables
¯layers
°layer_metrics
±non_trainable_variables
¹	variables
ºregularization_losses
 ²layer_regularization_losses
³__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
<
_0
`1
k2
l3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
³metrics
¾trainable_variables
´layers
µlayer_metrics
¶non_trainable_variables
¿	variables
Àregularization_losses
 ·layer_regularization_losses
µ__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¸metrics
Ãtrainable_variables
¹layers
ºlayer_metrics
»non_trainable_variables
Ä	variables
Åregularization_losses
 ¼layer_regularization_losses
·__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
.0
/1
02
13"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
½metrics
×trainable_variables
¾layers
¿layer_metrics
Ànon_trainable_variables
Ø	variables
Ùregularization_losses
 Álayer_regularization_losses
¹__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Âmetrics
Ütrainable_variables
Ãlayers
Älayer_metrics
Ånon_trainable_variables
Ý	variables
Þregularization_losses
 Ælayer_regularization_losses
»__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
<
c0
d1
m2
n3"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Çmetrics
âtrainable_variables
Èlayers
Élayer_metrics
Ênon_trainable_variables
ã	variables
äregularization_losses
 Ëlayer_regularization_losses
½__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ìmetrics
çtrainable_variables
Ílayers
Îlayer_metrics
Ïnon_trainable_variables
è	variables
éregularization_losses
 Ðlayer_regularization_losses
¿__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
>0
?1
@2
A3"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ñmetrics
ötrainable_variables
Òlayers
Ólayer_metrics
Ônon_trainable_variables
÷	variables
øregularization_losses
 Õlayer_regularization_losses
Á__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ömetrics
ûtrainable_variables
×layers
Ølayer_metrics
Ùnon_trainable_variables
ü	variables
ýregularization_losses
 Úlayer_regularization_losses
Ã__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿

Ûtotal

Ücount
Ý	variables
Þ	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


ßtotal

àcount
á
_fn_kwargs
â	variables
ã	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
Û0
Ü1"
trackable_list_wrapper
.
Ý	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ß0
à1"
trackable_list_wrapper
.
â	variables"
_generic_user_object
7:5 2Adam/sequential/conv2d/kernel/m
):' 2Adam/sequential/conv2d/bias/m
7:5 2+Adam/sequential/batch_normalization/gamma/m
6:4 2*Adam/sequential/batch_normalization/beta/m
;:9  2#Adam/sequential_1/conv2d_1/kernel/m
-:+ 2!Adam/sequential_1/conv2d_1/bias/m
;:9 2/Adam/sequential_1/batch_normalization_1/gamma/m
::8 2.Adam/sequential_1/batch_normalization_1/beta/m
;:9@@2#Adam/sequential_3/conv2d_3/kernel/m
-:+@2!Adam/sequential_3/conv2d_3/bias/m
;:9@2/Adam/sequential_3/batch_normalization_3/gamma/m
::8@2.Adam/sequential_3/batch_normalization_3/beta/m
=:;2#Adam/sequential_5/conv2d_5/kernel/m
.:,2!Adam/sequential_5/conv2d_5/bias/m
<::2/Adam/sequential_5/batch_normalization_5/gamma/m
;:92.Adam/sequential_5/batch_normalization_5/beta/m
2:0
2 Adam/sequential_6/dense/kernel/m
*:(2Adam/sequential_6/dense/bias/m
7:5 2Adam/sequential/conv2d/kernel/v
):' 2Adam/sequential/conv2d/bias/v
7:5 2+Adam/sequential/batch_normalization/gamma/v
6:4 2*Adam/sequential/batch_normalization/beta/v
;:9  2#Adam/sequential_1/conv2d_1/kernel/v
-:+ 2!Adam/sequential_1/conv2d_1/bias/v
;:9 2/Adam/sequential_1/batch_normalization_1/gamma/v
::8 2.Adam/sequential_1/batch_normalization_1/beta/v
;:9@@2#Adam/sequential_3/conv2d_3/kernel/v
-:+@2!Adam/sequential_3/conv2d_3/bias/v
;:9@2/Adam/sequential_3/batch_normalization_3/gamma/v
::8@2.Adam/sequential_3/batch_normalization_3/beta/v
=:;2#Adam/sequential_5/conv2d_5/kernel/v
.:,2!Adam/sequential_5/conv2d_5/bias/v
<::2/Adam/sequential_5/batch_normalization_5/gamma/v
;:92.Adam/sequential_5/batch_normalization_5/beta/v
2:0
2 Adam/sequential_6/dense/kernel/v
*:(2Adam/sequential_6/dense/bias/v
è2å
 __inference__wrapped_model_24101À
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿ
þ2û
,__inference_functional_1_layer_call_fn_26678
,__inference_functional_1_layer_call_fn_26043
,__inference_functional_1_layer_call_fn_26621
,__inference_functional_1_layer_call_fn_26167À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_functional_1_layer_call_and_return_conditional_losses_26564
G__inference_functional_1_layer_call_and_return_conditional_losses_26403
G__inference_functional_1_layer_call_and_return_conditional_losses_25918
G__inference_functional_1_layer_call_and_return_conditional_losses_25851À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
*__inference_sequential_layer_call_fn_26773
*__inference_sequential_layer_call_fn_24391
*__inference_sequential_layer_call_fn_26790
*__inference_sequential_layer_call_fn_24427À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
E__inference_sequential_layer_call_and_return_conditional_losses_24335
E__inference_sequential_layer_call_and_return_conditional_losses_26756
E__inference_sequential_layer_call_and_return_conditional_losses_26718
E__inference_sequential_layer_call_and_return_conditional_losses_24354À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
²2¯
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_24433à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_max_pooling2d_3_layer_call_fn_24439à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
þ2û
,__inference_sequential_1_layer_call_fn_24744
,__inference_sequential_1_layer_call_fn_26904
,__inference_sequential_1_layer_call_fn_26887
,__inference_sequential_1_layer_call_fn_24781À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_1_layer_call_and_return_conditional_losses_26831
G__inference_sequential_1_layer_call_and_return_conditional_losses_26870
G__inference_sequential_1_layer_call_and_return_conditional_losses_24686
G__inference_sequential_1_layer_call_and_return_conditional_losses_24706À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ð2í
F__inference_concatenate_layer_call_and_return_conditional_losses_26911¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Õ2Ò
+__inference_concatenate_layer_call_fn_26917¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
²2¯
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_24787à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_max_pooling2d_4_layer_call_fn_24793à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
þ2û
,__inference_sequential_3_layer_call_fn_25098
,__inference_sequential_3_layer_call_fn_27014
,__inference_sequential_3_layer_call_fn_27031
,__inference_sequential_3_layer_call_fn_25135À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_3_layer_call_and_return_conditional_losses_26997
G__inference_sequential_3_layer_call_and_return_conditional_losses_26958
G__inference_sequential_3_layer_call_and_return_conditional_losses_25060
G__inference_sequential_3_layer_call_and_return_conditional_losses_25040À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
H__inference_concatenate_1_layer_call_and_return_conditional_losses_27038¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_concatenate_1_layer_call_fn_27044¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
²2¯
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_25141à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_max_pooling2d_5_layer_call_fn_25147à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
þ2û
,__inference_sequential_5_layer_call_fn_27141
,__inference_sequential_5_layer_call_fn_27158
,__inference_sequential_5_layer_call_fn_25489
,__inference_sequential_5_layer_call_fn_25452À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_5_layer_call_and_return_conditional_losses_25394
G__inference_sequential_5_layer_call_and_return_conditional_losses_27085
G__inference_sequential_5_layer_call_and_return_conditional_losses_27124
G__inference_sequential_5_layer_call_and_return_conditional_losses_25414À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
H__inference_concatenate_2_layer_call_and_return_conditional_losses_27165¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
×2Ô
-__inference_concatenate_2_layer_call_fn_27171¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
þ2û
,__inference_sequential_6_layer_call_fn_27206
,__inference_sequential_6_layer_call_fn_27215
,__inference_sequential_6_layer_call_fn_25584
,__inference_sequential_6_layer_call_fn_25565À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ê2ç
G__inference_sequential_6_layer_call_and_return_conditional_losses_25535
G__inference_sequential_6_layer_call_and_return_conditional_losses_27184
G__inference_sequential_6_layer_call_and_return_conditional_losses_27197
G__inference_sequential_6_layer_call_and_return_conditional_losses_25545À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2B0
#__inference_signature_wrapper_26234input_1
ë2è
A__inference_conv2d_layer_call_and_return_conditional_losses_27225¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ð2Í
&__inference_conv2d_layer_call_fn_27234¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_1_layer_call_and_return_conditional_losses_27252¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_1_layer_call_fn_27257¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27295
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27359
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27341
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27277´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
3__inference_batch_normalization_layer_call_fn_27308
3__inference_batch_normalization_layer_call_fn_27321
3__inference_batch_normalization_layer_call_fn_27372
3__inference_batch_normalization_layer_call_fn_27385´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
í2ê
C__inference_conv2d_1_layer_call_and_return_conditional_losses_27395¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_conv2d_1_layer_call_fn_27404¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_2_layer_call_and_return_conditional_losses_27422¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_2_layer_call_fn_27427¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ÿ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27465
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27447
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27511
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27529´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
5__inference_batch_normalization_1_layer_call_fn_27555
5__inference_batch_normalization_1_layer_call_fn_27478
5__inference_batch_normalization_1_layer_call_fn_27491
5__inference_batch_normalization_1_layer_call_fn_27542´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
°2­
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_24549à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
-__inference_max_pooling2d_layer_call_fn_24555à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
í2ê
C__inference_conv2d_3_layer_call_and_return_conditional_losses_27565¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_conv2d_3_layer_call_fn_27574¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_4_layer_call_and_return_conditional_losses_27592¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_4_layer_call_fn_27597¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ÿ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27617
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27635
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27681
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27699´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
5__inference_batch_normalization_3_layer_call_fn_27712
5__inference_batch_normalization_3_layer_call_fn_27648
5__inference_batch_normalization_3_layer_call_fn_27725
5__inference_batch_normalization_3_layer_call_fn_27661´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
²2¯
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_24903à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_max_pooling2d_1_layer_call_fn_24909à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
í2ê
C__inference_conv2d_5_layer_call_and_return_conditional_losses_27735¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_conv2d_5_layer_call_fn_27744¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_activation_6_layer_call_and_return_conditional_losses_27762¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_activation_6_layer_call_fn_27767¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2ÿ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27787
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27851
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27869
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27805´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
5__inference_batch_normalization_5_layer_call_fn_27818
5__inference_batch_normalization_5_layer_call_fn_27831
5__inference_batch_normalization_5_layer_call_fn_27882
5__inference_batch_normalization_5_layer_call_fn_27895´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
²2¯
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_25257à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
/__inference_max_pooling2d_2_layer_call_fn_25263à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ì2é
B__inference_flatten_layer_call_and_return_conditional_losses_27901¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_flatten_layer_call_fn_27906¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_27917¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ï2Ì
%__inference_dense_layer_call_fn_27926¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 º
 __inference__wrapped_model_24101UVWXghYZ[\ij]^_`klabcdmnef:¢7
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿ
ª ";ª8
6
sequential_6&#
sequential_6ÿÿÿÿÿÿÿÿÿ·
G__inference_activation_1_layer_call_and_return_conditional_losses_27252l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_activation_1_layer_call_fn_27257_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿ ·
G__inference_activation_2_layer_call_and_return_conditional_losses_27422l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_activation_2_layer_call_fn_27427_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿ ·
G__inference_activation_4_layer_call_and_return_conditional_losses_27592l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_activation_4_layer_call_fn_27597_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@µ
G__inference_activation_6_layer_call_and_return_conditional_losses_27762j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
,__inference_activation_6_layer_call_fn_27767]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@ë
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27447[\ijM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ë
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27465[\ijM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ê
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27511v[\ij=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Ê
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_27529v[\ij=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Ã
5__inference_batch_normalization_1_layer_call_fn_27478[\ijM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ã
5__inference_batch_normalization_1_layer_call_fn_27491[\ijM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¢
5__inference_batch_normalization_1_layer_call_fn_27542i[\ij=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p
ª ""ÿÿÿÿÿÿÿÿÿ ¢
5__inference_batch_normalization_1_layer_call_fn_27555i[\ij=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª ""ÿÿÿÿÿÿÿÿÿ ë
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27617_`klM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ë
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27635_`klM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ê
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27681v_`kl=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 Ê
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_27699v_`kl=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 Ã
5__inference_batch_normalization_3_layer_call_fn_27648_`klM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ã
5__inference_batch_normalization_3_layer_call_fn_27661_`klM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¢
5__inference_batch_normalization_3_layer_call_fn_27712i_`kl=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ@
p
ª ""ÿÿÿÿÿÿÿÿÿ@¢
5__inference_batch_normalization_3_layer_call_fn_27725i_`kl=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª ""ÿÿÿÿÿÿÿÿÿ@í
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27787cdmnN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 í
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27805cdmnN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27851tcdmn<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 È
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27869tcdmn<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 Å
5__inference_batch_normalization_5_layer_call_fn_27818cdmnN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÅ
5__inference_batch_normalization_5_layer_call_fn_27831cdmnN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
5__inference_batch_normalization_5_layer_call_fn_27882gcdmn<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p
ª "!ÿÿÿÿÿÿÿÿÿ@@ 
5__inference_batch_normalization_5_layer_call_fn_27895gcdmn<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p 
ª "!ÿÿÿÿÿÿÿÿÿ@@é
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27277WXghM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 é
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27295WXghM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 È
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27341vWXgh=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 È
N__inference_batch_normalization_layer_call_and_return_conditional_losses_27359vWXgh=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Á
3__inference_batch_normalization_layer_call_fn_27308WXghM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Á
3__inference_batch_normalization_layer_call_fn_27321WXghM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ  
3__inference_batch_normalization_layer_call_fn_27372iWXgh=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p
ª ""ÿÿÿÿÿÿÿÿÿ  
3__inference_batch_normalization_layer_call_fn_27385iWXgh=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª ""ÿÿÿÿÿÿÿÿÿ é
H__inference_concatenate_1_layer_call_and_return_conditional_losses_27038j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ@@@
*'
inputs/1ÿÿÿÿÿÿÿÿÿ@@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 Á
-__inference_concatenate_1_layer_call_fn_27044j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ@@@
*'
inputs/1ÿÿÿÿÿÿÿÿÿ@@@
ª "!ÿÿÿÿÿÿÿÿÿ@@ë
H__inference_concatenate_2_layer_call_and_return_conditional_losses_27165l¢i
b¢_
]Z
+(
inputs/0ÿÿÿÿÿÿÿÿÿ  
+(
inputs/1ÿÿÿÿÿÿÿÿÿ  
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 Ã
-__inference_concatenate_2_layer_call_fn_27171l¢i
b¢_
]Z
+(
inputs/0ÿÿÿÿÿÿÿÿÿ  
+(
inputs/1ÿÿÿÿÿÿÿÿÿ  
ª "!ÿÿÿÿÿÿÿÿÿ  ì
F__inference_concatenate_layer_call_and_return_conditional_losses_26911¡n¢k
d¢a
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿ 
,)
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 Ä
+__inference_concatenate_layer_call_fn_26917n¢k
d¢a
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿ 
,)
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿ@·
C__inference_conv2d_1_layer_call_and_return_conditional_losses_27395pYZ9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
(__inference_conv2d_1_layer_call_fn_27404cYZ9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿ ·
C__inference_conv2d_3_layer_call_and_return_conditional_losses_27565p]^9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
(__inference_conv2d_3_layer_call_fn_27574c]^9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@µ
C__inference_conv2d_5_layer_call_and_return_conditional_losses_27735nab8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
(__inference_conv2d_5_layer_call_fn_27744aab8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@µ
A__inference_conv2d_layer_call_and_return_conditional_losses_27225pUV9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
&__inference_conv2d_layer_call_fn_27234cUV9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ ¢
@__inference_dense_layer_call_and_return_conditional_losses_27917^ef1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
%__inference_dense_layer_call_fn_27926Qef1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
B__inference_flatten_layer_call_and_return_conditional_losses_27901c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 
'__inference_flatten_layer_call_fn_27906V8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª "ÿÿÿÿÿÿÿÿÿÓ
G__inference_functional_1_layer_call_and_return_conditional_losses_25851UVWXghYZ[\ij]^_`klabcdmnefB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ó
G__inference_functional_1_layer_call_and_return_conditional_losses_25918UVWXghYZ[\ij]^_`klabcdmnefB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ò
G__inference_functional_1_layer_call_and_return_conditional_losses_26403UVWXghYZ[\ij]^_`klabcdmnefA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ò
G__inference_functional_1_layer_call_and_return_conditional_losses_26564UVWXghYZ[\ij]^_`klabcdmnefA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ª
,__inference_functional_1_layer_call_fn_26043zUVWXghYZ[\ij]^_`klabcdmnefB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿª
,__inference_functional_1_layer_call_fn_26167zUVWXghYZ[\ij]^_`klabcdmnefB¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ©
,__inference_functional_1_layer_call_fn_26621yUVWXghYZ[\ij]^_`klabcdmnefA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ©
,__inference_functional_1_layer_call_fn_26678yUVWXghYZ[\ij]^_`klabcdmnefA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_24903R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_1_layer_call_fn_24909R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_25257R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_2_layer_call_fn_25263R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_24433R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_3_layer_call_fn_24439R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_24787R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_4_layer_call_fn_24793R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_25141R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_5_layer_call_fn_25147R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_24549R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_max_pooling2d_layer_call_fn_24555R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
G__inference_sequential_1_layer_call_and_return_conditional_losses_24686YZ[\ijI¢F
?¢<
2/
conv2d_1_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Ð
G__inference_sequential_1_layer_call_and_return_conditional_losses_24706YZ[\ijI¢F
?¢<
2/
conv2d_1_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Ç
G__inference_sequential_1_layer_call_and_return_conditional_losses_26831|YZ[\ijA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Ç
G__inference_sequential_1_layer_call_and_return_conditional_losses_26870|YZ[\ijA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 §
,__inference_sequential_1_layer_call_fn_24744wYZ[\ijI¢F
?¢<
2/
conv2d_1_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª ""ÿÿÿÿÿÿÿÿÿ §
,__inference_sequential_1_layer_call_fn_24781wYZ[\ijI¢F
?¢<
2/
conv2d_1_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ 
,__inference_sequential_1_layer_call_fn_26887oYZ[\ijA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª ""ÿÿÿÿÿÿÿÿÿ 
,__inference_sequential_1_layer_call_fn_26904oYZ[\ijA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ Î
G__inference_sequential_3_layer_call_and_return_conditional_losses_25040]^_`klI¢F
?¢<
2/
conv2d_3_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 Î
G__inference_sequential_3_layer_call_and_return_conditional_losses_25060]^_`klI¢F
?¢<
2/
conv2d_3_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 Å
G__inference_sequential_3_layer_call_and_return_conditional_losses_26958z]^_`klA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 Å
G__inference_sequential_3_layer_call_and_return_conditional_losses_26997z]^_`klA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@@@
 ¥
,__inference_sequential_3_layer_call_fn_25098u]^_`klI¢F
?¢<
2/
conv2d_3_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª " ÿÿÿÿÿÿÿÿÿ@@@¥
,__inference_sequential_3_layer_call_fn_25135u]^_`klI¢F
?¢<
2/
conv2d_3_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª " ÿÿÿÿÿÿÿÿÿ@@@
,__inference_sequential_3_layer_call_fn_27014m]^_`klA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª " ÿÿÿÿÿÿÿÿÿ@@@
,__inference_sequential_3_layer_call_fn_27031m]^_`klA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª " ÿÿÿÿÿÿÿÿÿ@@@Î
G__inference_sequential_5_layer_call_and_return_conditional_losses_25394abcdmnH¢E
>¢;
1.
conv2d_5_inputÿÿÿÿÿÿÿÿÿ@@
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 Î
G__inference_sequential_5_layer_call_and_return_conditional_losses_25414abcdmnH¢E
>¢;
1.
conv2d_5_inputÿÿÿÿÿÿÿÿÿ@@
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 Å
G__inference_sequential_5_layer_call_and_return_conditional_losses_27085zabcdmn@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 Å
G__inference_sequential_5_layer_call_and_return_conditional_losses_27124zabcdmn@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 ¥
,__inference_sequential_5_layer_call_fn_25452uabcdmnH¢E
>¢;
1.
conv2d_5_inputÿÿÿÿÿÿÿÿÿ@@
p

 
ª "!ÿÿÿÿÿÿÿÿÿ  ¥
,__inference_sequential_5_layer_call_fn_25489uabcdmnH¢E
>¢;
1.
conv2d_5_inputÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ  
,__inference_sequential_5_layer_call_fn_27141mabcdmn@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p

 
ª "!ÿÿÿÿÿÿÿÿÿ  
,__inference_sequential_5_layer_call_fn_27158mabcdmn@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ  ¿
G__inference_sequential_6_layer_call_and_return_conditional_losses_25535tefG¢D
=¢:
0-
flatten_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
G__inference_sequential_6_layer_call_and_return_conditional_losses_25545tefG¢D
=¢:
0-
flatten_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
G__inference_sequential_6_layer_call_and_return_conditional_losses_27184mef@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¸
G__inference_sequential_6_layer_call_and_return_conditional_losses_27197mef@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_sequential_6_layer_call_fn_25565gefG¢D
=¢:
0-
flatten_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_6_layer_call_fn_25584gefG¢D
=¢:
0-
flatten_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_6_layer_call_fn_27206`ef@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_6_layer_call_fn_27215`ef@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿÌ
E__inference_sequential_layer_call_and_return_conditional_losses_24335UVWXghG¢D
=¢:
0-
conv2d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Ì
E__inference_sequential_layer_call_and_return_conditional_losses_24354UVWXghG¢D
=¢:
0-
conv2d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Å
E__inference_sequential_layer_call_and_return_conditional_losses_26718|UVWXghA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Å
E__inference_sequential_layer_call_and_return_conditional_losses_26756|UVWXghA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 £
*__inference_sequential_layer_call_fn_24391uUVWXghG¢D
=¢:
0-
conv2d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿ £
*__inference_sequential_layer_call_fn_24427uUVWXghG¢D
=¢:
0-
conv2d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ 
*__inference_sequential_layer_call_fn_26773oUVWXghA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿ 
*__inference_sequential_layer_call_fn_26790oUVWXghA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ È
#__inference_signature_wrapper_26234 UVWXghYZ[\ij]^_`klabcdmnefE¢B
¢ 
;ª8
6
input_1+(
input_1ÿÿÿÿÿÿÿÿÿ";ª8
6
sequential_6&#
sequential_6ÿÿÿÿÿÿÿÿÿ