Ò3
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
 "serve*2.3.02v2.3.0-0-gb36436b8Î(
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

sequential_2/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*-
shared_namesequential_2/conv2d_2/kernel

0sequential_2/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpsequential_2/conv2d_2/kernel*&
_output_shapes
:@@*
dtype0

sequential_2/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namesequential_2/conv2d_2/bias

.sequential_2/conv2d_2/bias/Read/ReadVariableOpReadVariableOpsequential_2/conv2d_2/bias*
_output_shapes
:@*
dtype0
¨
(sequential_2/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*9
shared_name*(sequential_2/batch_normalization_2/gamma
¡
<sequential_2/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp(sequential_2/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
¦
'sequential_2/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'sequential_2/batch_normalization_2/beta

;sequential_2/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp'sequential_2/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
´
.sequential_2/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.sequential_2/batch_normalization_2/moving_mean
­
Bsequential_2/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp.sequential_2/batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
¼
2sequential_2/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*C
shared_name42sequential_2/batch_normalization_2/moving_variance
µ
Fsequential_2/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp2sequential_2/batch_normalization_2/moving_variance*
_output_shapes
:@*
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

sequential_4/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namesequential_4/conv2d_4/kernel

0sequential_4/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpsequential_4/conv2d_4/kernel*(
_output_shapes
:*
dtype0

sequential_4/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namesequential_4/conv2d_4/bias

.sequential_4/conv2d_4/bias/Read/ReadVariableOpReadVariableOpsequential_4/conv2d_4/bias*
_output_shapes	
:*
dtype0
©
(sequential_4/batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(sequential_4/batch_normalization_4/gamma
¢
<sequential_4/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOp(sequential_4/batch_normalization_4/gamma*
_output_shapes	
:*
dtype0
§
'sequential_4/batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'sequential_4/batch_normalization_4/beta
 
;sequential_4/batch_normalization_4/beta/Read/ReadVariableOpReadVariableOp'sequential_4/batch_normalization_4/beta*
_output_shapes	
:*
dtype0
µ
.sequential_4/batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.sequential_4/batch_normalization_4/moving_mean
®
Bsequential_4/batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp.sequential_4/batch_normalization_4/moving_mean*
_output_shapes	
:*
dtype0
½
2sequential_4/batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42sequential_4/batch_normalization_4/moving_variance
¶
Fsequential_4/batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp2sequential_4/batch_normalization_4/moving_variance*
_output_shapes	
:*
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
#Adam/sequential_2/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#Adam/sequential_2/conv2d_2/kernel/m
£
7Adam/sequential_2/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_2/conv2d_2/kernel/m*&
_output_shapes
:@@*
dtype0

!Adam/sequential_2/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/sequential_2/conv2d_2/bias/m

5Adam/sequential_2/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_2/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
¶
/Adam/sequential_2/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/Adam/sequential_2/batch_normalization_2/gamma/m
¯
CAdam/sequential_2/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp/Adam/sequential_2/batch_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
´
.Adam/sequential_2/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/sequential_2/batch_normalization_2/beta/m
­
BAdam/sequential_2/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp.Adam/sequential_2/batch_normalization_2/beta/m*
_output_shapes
:@*
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
#Adam/sequential_4/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/sequential_4/conv2d_4/kernel/m
¥
7Adam/sequential_4/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_4/conv2d_4/kernel/m*(
_output_shapes
:*
dtype0

!Adam/sequential_4/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/sequential_4/conv2d_4/bias/m

5Adam/sequential_4/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_4/conv2d_4/bias/m*
_output_shapes	
:*
dtype0
·
/Adam/sequential_4/batch_normalization_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adam/sequential_4/batch_normalization_4/gamma/m
°
CAdam/sequential_4/batch_normalization_4/gamma/m/Read/ReadVariableOpReadVariableOp/Adam/sequential_4/batch_normalization_4/gamma/m*
_output_shapes	
:*
dtype0
µ
.Adam/sequential_4/batch_normalization_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/sequential_4/batch_normalization_4/beta/m
®
BAdam/sequential_4/batch_normalization_4/beta/m/Read/ReadVariableOpReadVariableOp.Adam/sequential_4/batch_normalization_4/beta/m*
_output_shapes	
:*
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
#Adam/sequential_2/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#Adam/sequential_2/conv2d_2/kernel/v
£
7Adam/sequential_2/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_2/conv2d_2/kernel/v*&
_output_shapes
:@@*
dtype0

!Adam/sequential_2/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/sequential_2/conv2d_2/bias/v

5Adam/sequential_2/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_2/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
¶
/Adam/sequential_2/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*@
shared_name1/Adam/sequential_2/batch_normalization_2/gamma/v
¯
CAdam/sequential_2/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp/Adam/sequential_2/batch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
´
.Adam/sequential_2/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.Adam/sequential_2/batch_normalization_2/beta/v
­
BAdam/sequential_2/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp.Adam/sequential_2/batch_normalization_2/beta/v*
_output_shapes
:@*
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
#Adam/sequential_4/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/sequential_4/conv2d_4/kernel/v
¥
7Adam/sequential_4/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_4/conv2d_4/kernel/v*(
_output_shapes
:*
dtype0

!Adam/sequential_4/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/sequential_4/conv2d_4/bias/v

5Adam/sequential_4/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_4/conv2d_4/bias/v*
_output_shapes	
:*
dtype0
·
/Adam/sequential_4/batch_normalization_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*@
shared_name1/Adam/sequential_4/batch_normalization_4/gamma/v
°
CAdam/sequential_4/batch_normalization_4/gamma/v/Read/ReadVariableOpReadVariableOp/Adam/sequential_4/batch_normalization_4/gamma/v*
_output_shapes	
:*
dtype0
µ
.Adam/sequential_4/batch_normalization_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.Adam/sequential_4/batch_normalization_4/beta/v
®
BAdam/sequential_4/batch_normalization_4/beta/v/Read/ReadVariableOpReadVariableOp.Adam/sequential_4/batch_normalization_4/beta/v*
_output_shapes	
:*
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
³Ò
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*íÑ
valueâÑBÞÑ BÖÑ
á
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
­
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
º
 layer_with_weights-0
 layer-0
!layer-1
"layer_with_weights-1
"layer-2
#layer-3
$	variables
%trainable_variables
&regularization_losses
'	keras_api
R
(	variables
)trainable_variables
*regularization_losses
+	keras_api
­
,layer_with_weights-0
,layer-0
-layer-1
.layer_with_weights-1
.layer-2
/	variables
0trainable_variables
1regularization_losses
2	keras_api
R
3	variables
4trainable_variables
5regularization_losses
6	keras_api
º
7layer_with_weights-0
7layer-0
8layer-1
9layer_with_weights-1
9layer-2
:layer-3
;	variables
<trainable_variables
=regularization_losses
>	keras_api
R
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
­
Clayer_with_weights-0
Clayer-0
Dlayer-1
Elayer_with_weights-1
Elayer-2
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
R
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
º
Nlayer_with_weights-0
Nlayer-0
Olayer-1
Player_with_weights-1
Player-2
Qlayer-3
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
R
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api

Zlayer-0
[layer_with_weights-0
[layer-1
\	variables
]trainable_variables
^regularization_losses
_	keras_api
Ö
`iter

abeta_1

bbeta_2
	cdecay
dlearning_rateemÈfmÉgmÊhmËkmÌlmÍmmÎnmÏqmÐrmÑsmÒtmÓwmÔxmÕymÖzm×}mØ~mÙmÚ	mÛ	mÜ	mÝ	mÞ	mß	mà	máevâfvãgvähvåkvælvçmvènvéqvêrvësvìtvíwvîxvïyvðzvñ}vò~vóvô	võ	vö	v÷	vø	vù	vú	vû
±
e0
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
r13
s14
t15
u16
v17
w18
x19
y20
z21
{22
|23
}24
~25
26
27
28
29
30
31
32
33
34
35
36
37
Í
e0
f1
g2
h3
k4
l5
m6
n7
q8
r9
s10
t11
w12
x13
y14
z15
}16
~17
18
19
20
21
22
23
24
25
 
²
layers
 layer_regularization_losses
	variables
layer_metrics
trainable_variables
regularization_losses
non_trainable_variables
metrics
 

_inbound_nodes

ekernel
fbias
	variables
trainable_variables
regularization_losses
	keras_api
k
_inbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
±
_inbound_nodes
	axis
	ggamma
hbeta
imoving_mean
jmoving_variance
	variables
trainable_variables
regularization_losses
	keras_api
*
e0
f1
g2
h3
i4
j5

e0
f1
g2
h3
 
²
 layers
 ¡layer_regularization_losses
	variables
¢layer_metrics
trainable_variables
regularization_losses
£non_trainable_variables
¤metrics
 
 
 
²
¥layers
 ¦layer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
¨non_trainable_variables
©metrics

ª_inbound_nodes

kkernel
lbias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
k
¯_inbound_nodes
°	variables
±trainable_variables
²regularization_losses
³	keras_api
±
´_inbound_nodes
	µaxis
	mgamma
nbeta
omoving_mean
pmoving_variance
¶	variables
·trainable_variables
¸regularization_losses
¹	keras_api
k
º_inbound_nodes
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
*
k0
l1
m2
n3
o4
p5

k0
l1
m2
n3
 
²
¿layers
 Àlayer_regularization_losses
$	variables
Álayer_metrics
%trainable_variables
&regularization_losses
Ânon_trainable_variables
Ãmetrics
 
 
 
²
Älayers
 Ålayer_regularization_losses
Ælayer_metrics
(	variables
)trainable_variables
*regularization_losses
Çnon_trainable_variables
Èmetrics

É_inbound_nodes

qkernel
rbias
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
k
Î_inbound_nodes
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api
±
Ó_inbound_nodes
	Ôaxis
	sgamma
tbeta
umoving_mean
vmoving_variance
Õ	variables
Ötrainable_variables
×regularization_losses
Ø	keras_api
*
q0
r1
s2
t3
u4
v5

q0
r1
s2
t3
 
²
Ùlayers
 Úlayer_regularization_losses
/	variables
Ûlayer_metrics
0trainable_variables
1regularization_losses
Ünon_trainable_variables
Ýmetrics
 
 
 
²
Þlayers
 ßlayer_regularization_losses
àlayer_metrics
3	variables
4trainable_variables
5regularization_losses
ánon_trainable_variables
âmetrics

ã_inbound_nodes

wkernel
xbias
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
k
è_inbound_nodes
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
±
í_inbound_nodes
	îaxis
	ygamma
zbeta
{moving_mean
|moving_variance
ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
k
ó_inbound_nodes
ô	variables
õtrainable_variables
öregularization_losses
÷	keras_api
*
w0
x1
y2
z3
{4
|5

w0
x1
y2
z3
 
²
ølayers
 ùlayer_regularization_losses
;	variables
úlayer_metrics
<trainable_variables
=regularization_losses
ûnon_trainable_variables
ümetrics
 
 
 
²
ýlayers
 þlayer_regularization_losses
ÿlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
non_trainable_variables
metrics

_inbound_nodes

}kernel
~bias
	variables
trainable_variables
regularization_losses
	keras_api
k
_inbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
´
_inbound_nodes
	axis
	gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
-
}0
~1
2
3
4
5

}0
~1
2
3
 
²
layers
 layer_regularization_losses
F	variables
layer_metrics
Gtrainable_variables
Hregularization_losses
non_trainable_variables
metrics
 
 
 
²
layers
 layer_regularization_losses
layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
non_trainable_variables
metrics

_inbound_nodes
kernel
	bias
	variables
trainable_variables
regularization_losses
 	keras_api
k
¡_inbound_nodes
¢	variables
£trainable_variables
¤regularization_losses
¥	keras_api
µ
¦_inbound_nodes
	§axis

gamma
	beta
moving_mean
moving_variance
¨	variables
©trainable_variables
ªregularization_losses
«	keras_api
k
¬_inbound_nodes
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
0
0
1
2
3
4
5
 
0
1
2
3
 
²
±layers
 ²layer_regularization_losses
R	variables
³layer_metrics
Strainable_variables
Tregularization_losses
´non_trainable_variables
µmetrics
 
 
 
²
¶layers
 ·layer_regularization_losses
¸layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
¹non_trainable_variables
ºmetrics
k
»_inbound_nodes
¼	variables
½trainable_variables
¾regularization_losses
¿	keras_api

À_inbound_nodes
kernel
	bias
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api

0
1

0
1
 
²
Ålayers
 Ælayer_regularization_losses
\	variables
Çlayer_metrics
]trainable_variables
^regularization_losses
Ènon_trainable_variables
Émetrics
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
TR
VARIABLE_VALUEsequential/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEsequential/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE$sequential/batch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#sequential/batch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE*sequential/batch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE.sequential/batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEsequential_1/conv2d_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEsequential_1/conv2d_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE(sequential_1/batch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUE'sequential_1/batch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.sequential_1/batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2sequential_1/batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential_2/conv2d_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEsequential_2/conv2d_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(sequential_2/batch_normalization_2/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'sequential_2/batch_normalization_2/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.sequential_2/batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2sequential_2/batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential_3/conv2d_3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEsequential_3/conv2d_3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(sequential_3/batch_normalization_3/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'sequential_3/batch_normalization_3/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.sequential_3/batch_normalization_3/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2sequential_3/batch_normalization_3/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential_4/conv2d_4/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEsequential_4/conv2d_4/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(sequential_4/batch_normalization_4/gamma'variables/26/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'sequential_4/batch_normalization_4/beta'variables/27/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.sequential_4/batch_normalization_4/moving_mean'variables/28/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2sequential_4/batch_normalization_4/moving_variance'variables/29/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEsequential_5/conv2d_5/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEsequential_5/conv2d_5/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUE(sequential_5/batch_normalization_5/gamma'variables/32/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'sequential_5/batch_normalization_5/beta'variables/33/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE.sequential_5/batch_normalization_5/moving_mean'variables/34/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE2sequential_5/batch_normalization_5/moving_variance'variables/35/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEsequential_6/dense/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEsequential_6/dense/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE
f
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
12
13
 
 
Z
i0
j1
o2
p3
u4
v5
{6
|7
8
9
10
11

Ê0
Ë1
 

e0
f1

e0
f1
 
µ
Ìlayers
 Ílayer_regularization_losses
Îlayer_metrics
	variables
trainable_variables
regularization_losses
Ïnon_trainable_variables
Ðmetrics
 
 
 
 
µ
Ñlayers
 Òlayer_regularization_losses
Ólayer_metrics
	variables
trainable_variables
regularization_losses
Ônon_trainable_variables
Õmetrics
 
 

g0
h1
i2
j3

g0
h1
 
µ
Ölayers
 ×layer_regularization_losses
Ølayer_metrics
	variables
trainable_variables
regularization_losses
Ùnon_trainable_variables
Úmetrics

0
1
2
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

k0
l1

k0
l1
 
µ
Ûlayers
 Ülayer_regularization_losses
Ýlayer_metrics
«	variables
¬trainable_variables
­regularization_losses
Þnon_trainable_variables
ßmetrics
 
 
 
 
µ
àlayers
 álayer_regularization_losses
âlayer_metrics
°	variables
±trainable_variables
²regularization_losses
ãnon_trainable_variables
ämetrics
 
 

m0
n1
o2
p3

m0
n1
 
µ
ålayers
 ælayer_regularization_losses
çlayer_metrics
¶	variables
·trainable_variables
¸regularization_losses
ènon_trainable_variables
émetrics
 
 
 
 
µ
êlayers
 ëlayer_regularization_losses
ìlayer_metrics
»	variables
¼trainable_variables
½regularization_losses
ínon_trainable_variables
îmetrics

 0
!1
"2
#3
 
 

o0
p1
 
 
 
 
 
 
 

q0
r1

q0
r1
 
µ
ïlayers
 ðlayer_regularization_losses
ñlayer_metrics
Ê	variables
Ëtrainable_variables
Ìregularization_losses
ònon_trainable_variables
ómetrics
 
 
 
 
µ
ôlayers
 õlayer_regularization_losses
ölayer_metrics
Ï	variables
Ðtrainable_variables
Ñregularization_losses
÷non_trainable_variables
ømetrics
 
 

s0
t1
u2
v3

s0
t1
 
µ
ùlayers
 úlayer_regularization_losses
ûlayer_metrics
Õ	variables
Ötrainable_variables
×regularization_losses
ünon_trainable_variables
ýmetrics

,0
-1
.2
 
 

u0
v1
 
 
 
 
 
 
 

w0
x1

w0
x1
 
µ
þlayers
 ÿlayer_regularization_losses
layer_metrics
ä	variables
åtrainable_variables
æregularization_losses
non_trainable_variables
metrics
 
 
 
 
µ
layers
 layer_regularization_losses
layer_metrics
é	variables
êtrainable_variables
ëregularization_losses
non_trainable_variables
metrics
 
 

y0
z1
{2
|3

y0
z1
 
µ
layers
 layer_regularization_losses
layer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
non_trainable_variables
metrics
 
 
 
 
µ
layers
 layer_regularization_losses
layer_metrics
ô	variables
õtrainable_variables
öregularization_losses
non_trainable_variables
metrics

70
81
92
:3
 
 

{0
|1
 
 
 
 
 
 
 

}0
~1

}0
~1
 
µ
layers
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
non_trainable_variables
metrics
 
 
 
 
µ
layers
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
non_trainable_variables
metrics
 
 

0
1
2
3

0
1
 
µ
layers
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
non_trainable_variables
 metrics

C0
D1
E2
 
 

0
1
 
 
 
 
 
 
 

0
1

0
1
 
µ
¡layers
 ¢layer_regularization_losses
£layer_metrics
	variables
trainable_variables
regularization_losses
¤non_trainable_variables
¥metrics
 
 
 
 
µ
¦layers
 §layer_regularization_losses
¨layer_metrics
¢	variables
£trainable_variables
¤regularization_losses
©non_trainable_variables
ªmetrics
 
 
 
0
1
2
3

0
1
 
µ
«layers
 ¬layer_regularization_losses
­layer_metrics
¨	variables
©trainable_variables
ªregularization_losses
®non_trainable_variables
¯metrics
 
 
 
 
µ
°layers
 ±layer_regularization_losses
²layer_metrics
­	variables
®trainable_variables
¯regularization_losses
³non_trainable_variables
´metrics

N0
O1
P2
Q3
 
 

0
1
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
µlayers
 ¶layer_regularization_losses
·layer_metrics
¼	variables
½trainable_variables
¾regularization_losses
¸non_trainable_variables
¹metrics
 

0
1

0
1
 
µ
ºlayers
 »layer_regularization_losses
¼layer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
½non_trainable_variables
¾metrics

Z0
[1
 
 
 
 
8

¿total

Àcount
Á	variables
Â	keras_api
I

Ãtotal

Äcount
Å
_fn_kwargs
Æ	variables
Ç	keras_api
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

o0
p1
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
u0
v1
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
{0
|1
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

0
1
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

0
1
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
¿0
À1

Á	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ã0
Ä1

Æ	variables
wu
VARIABLE_VALUEAdam/sequential/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/sequential/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+Adam/sequential/batch_normalization/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/sequential/batch_normalization/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/sequential_1/conv2d_1/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/sequential_1/conv2d_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_1/batch_normalization_1/gamma/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_1/batch_normalization_1/beta/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_2/conv2d_2/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/sequential_2/conv2d_2/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_2/batch_normalization_2/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_2/batch_normalization_2/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_3/conv2d_3/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/sequential_3/conv2d_3/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_3/batch_normalization_3/gamma/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_3/batch_normalization_3/beta/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_4/conv2d_4/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/sequential_4/conv2d_4/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_4/batch_normalization_4/gamma/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_4/batch_normalization_4/beta/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_5/conv2d_5/kernel/mCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/sequential_5/conv2d_5/bias/mCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_5/batch_normalization_5/gamma/mCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_5/batch_normalization_5/beta/mCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/sequential_6/dense/kernel/mCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential_6/dense/bias/mCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/sequential/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE+Adam/sequential/batch_normalization/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/sequential/batch_normalization/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/sequential_1/conv2d_1/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/sequential_1/conv2d_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_1/batch_normalization_1/gamma/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_1/batch_normalization_1/beta/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_2/conv2d_2/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/sequential_2/conv2d_2/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_2/batch_normalization_2/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_2/batch_normalization_2/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_3/conv2d_3/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/sequential_3/conv2d_3/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_3/batch_normalization_3/gamma/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_3/batch_normalization_3/beta/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_4/conv2d_4/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/sequential_4/conv2d_4/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_4/batch_normalization_4/gamma/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_4/batch_normalization_4/beta/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUE#Adam/sequential_5/conv2d_5/kernel/vCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/sequential_5/conv2d_5/bias/vCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE/Adam/sequential_5/batch_normalization_5/gamma/vCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE.Adam/sequential_5/batch_normalization_5/beta/vCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE Adam/sequential_6/dense/kernel/vCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential_6/dense/bias/vCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
Â
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1sequential/conv2d/kernelsequential/conv2d/bias$sequential/batch_normalization/gamma#sequential/batch_normalization/beta*sequential/batch_normalization/moving_mean.sequential/batch_normalization/moving_variancesequential_1/conv2d_1/kernelsequential_1/conv2d_1/bias(sequential_1/batch_normalization_1/gamma'sequential_1/batch_normalization_1/beta.sequential_1/batch_normalization_1/moving_mean2sequential_1/batch_normalization_1/moving_variancesequential_2/conv2d_2/kernelsequential_2/conv2d_2/bias(sequential_2/batch_normalization_2/gamma'sequential_2/batch_normalization_2/beta.sequential_2/batch_normalization_2/moving_mean2sequential_2/batch_normalization_2/moving_variancesequential_3/conv2d_3/kernelsequential_3/conv2d_3/bias(sequential_3/batch_normalization_3/gamma'sequential_3/batch_normalization_3/beta.sequential_3/batch_normalization_3/moving_mean2sequential_3/batch_normalization_3/moving_variancesequential_4/conv2d_4/kernelsequential_4/conv2d_4/bias(sequential_4/batch_normalization_4/gamma'sequential_4/batch_normalization_4/beta.sequential_4/batch_normalization_4/moving_mean2sequential_4/batch_normalization_4/moving_variancesequential_5/conv2d_5/kernelsequential_5/conv2d_5/bias(sequential_5/batch_normalization_5/gamma'sequential_5/batch_normalization_5/beta.sequential_5/batch_normalization_5/moving_mean2sequential_5/batch_normalization_5/moving_variancesequential_6/dense/kernelsequential_6/dense/bias*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_28143
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
³/
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp,sequential/conv2d/kernel/Read/ReadVariableOp*sequential/conv2d/bias/Read/ReadVariableOp8sequential/batch_normalization/gamma/Read/ReadVariableOp7sequential/batch_normalization/beta/Read/ReadVariableOp>sequential/batch_normalization/moving_mean/Read/ReadVariableOpBsequential/batch_normalization/moving_variance/Read/ReadVariableOp0sequential_1/conv2d_1/kernel/Read/ReadVariableOp.sequential_1/conv2d_1/bias/Read/ReadVariableOp<sequential_1/batch_normalization_1/gamma/Read/ReadVariableOp;sequential_1/batch_normalization_1/beta/Read/ReadVariableOpBsequential_1/batch_normalization_1/moving_mean/Read/ReadVariableOpFsequential_1/batch_normalization_1/moving_variance/Read/ReadVariableOp0sequential_2/conv2d_2/kernel/Read/ReadVariableOp.sequential_2/conv2d_2/bias/Read/ReadVariableOp<sequential_2/batch_normalization_2/gamma/Read/ReadVariableOp;sequential_2/batch_normalization_2/beta/Read/ReadVariableOpBsequential_2/batch_normalization_2/moving_mean/Read/ReadVariableOpFsequential_2/batch_normalization_2/moving_variance/Read/ReadVariableOp0sequential_3/conv2d_3/kernel/Read/ReadVariableOp.sequential_3/conv2d_3/bias/Read/ReadVariableOp<sequential_3/batch_normalization_3/gamma/Read/ReadVariableOp;sequential_3/batch_normalization_3/beta/Read/ReadVariableOpBsequential_3/batch_normalization_3/moving_mean/Read/ReadVariableOpFsequential_3/batch_normalization_3/moving_variance/Read/ReadVariableOp0sequential_4/conv2d_4/kernel/Read/ReadVariableOp.sequential_4/conv2d_4/bias/Read/ReadVariableOp<sequential_4/batch_normalization_4/gamma/Read/ReadVariableOp;sequential_4/batch_normalization_4/beta/Read/ReadVariableOpBsequential_4/batch_normalization_4/moving_mean/Read/ReadVariableOpFsequential_4/batch_normalization_4/moving_variance/Read/ReadVariableOp0sequential_5/conv2d_5/kernel/Read/ReadVariableOp.sequential_5/conv2d_5/bias/Read/ReadVariableOp<sequential_5/batch_normalization_5/gamma/Read/ReadVariableOp;sequential_5/batch_normalization_5/beta/Read/ReadVariableOpBsequential_5/batch_normalization_5/moving_mean/Read/ReadVariableOpFsequential_5/batch_normalization_5/moving_variance/Read/ReadVariableOp-sequential_6/dense/kernel/Read/ReadVariableOp+sequential_6/dense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp3Adam/sequential/conv2d/kernel/m/Read/ReadVariableOp1Adam/sequential/conv2d/bias/m/Read/ReadVariableOp?Adam/sequential/batch_normalization/gamma/m/Read/ReadVariableOp>Adam/sequential/batch_normalization/beta/m/Read/ReadVariableOp7Adam/sequential_1/conv2d_1/kernel/m/Read/ReadVariableOp5Adam/sequential_1/conv2d_1/bias/m/Read/ReadVariableOpCAdam/sequential_1/batch_normalization_1/gamma/m/Read/ReadVariableOpBAdam/sequential_1/batch_normalization_1/beta/m/Read/ReadVariableOp7Adam/sequential_2/conv2d_2/kernel/m/Read/ReadVariableOp5Adam/sequential_2/conv2d_2/bias/m/Read/ReadVariableOpCAdam/sequential_2/batch_normalization_2/gamma/m/Read/ReadVariableOpBAdam/sequential_2/batch_normalization_2/beta/m/Read/ReadVariableOp7Adam/sequential_3/conv2d_3/kernel/m/Read/ReadVariableOp5Adam/sequential_3/conv2d_3/bias/m/Read/ReadVariableOpCAdam/sequential_3/batch_normalization_3/gamma/m/Read/ReadVariableOpBAdam/sequential_3/batch_normalization_3/beta/m/Read/ReadVariableOp7Adam/sequential_4/conv2d_4/kernel/m/Read/ReadVariableOp5Adam/sequential_4/conv2d_4/bias/m/Read/ReadVariableOpCAdam/sequential_4/batch_normalization_4/gamma/m/Read/ReadVariableOpBAdam/sequential_4/batch_normalization_4/beta/m/Read/ReadVariableOp7Adam/sequential_5/conv2d_5/kernel/m/Read/ReadVariableOp5Adam/sequential_5/conv2d_5/bias/m/Read/ReadVariableOpCAdam/sequential_5/batch_normalization_5/gamma/m/Read/ReadVariableOpBAdam/sequential_5/batch_normalization_5/beta/m/Read/ReadVariableOp4Adam/sequential_6/dense/kernel/m/Read/ReadVariableOp2Adam/sequential_6/dense/bias/m/Read/ReadVariableOp3Adam/sequential/conv2d/kernel/v/Read/ReadVariableOp1Adam/sequential/conv2d/bias/v/Read/ReadVariableOp?Adam/sequential/batch_normalization/gamma/v/Read/ReadVariableOp>Adam/sequential/batch_normalization/beta/v/Read/ReadVariableOp7Adam/sequential_1/conv2d_1/kernel/v/Read/ReadVariableOp5Adam/sequential_1/conv2d_1/bias/v/Read/ReadVariableOpCAdam/sequential_1/batch_normalization_1/gamma/v/Read/ReadVariableOpBAdam/sequential_1/batch_normalization_1/beta/v/Read/ReadVariableOp7Adam/sequential_2/conv2d_2/kernel/v/Read/ReadVariableOp5Adam/sequential_2/conv2d_2/bias/v/Read/ReadVariableOpCAdam/sequential_2/batch_normalization_2/gamma/v/Read/ReadVariableOpBAdam/sequential_2/batch_normalization_2/beta/v/Read/ReadVariableOp7Adam/sequential_3/conv2d_3/kernel/v/Read/ReadVariableOp5Adam/sequential_3/conv2d_3/bias/v/Read/ReadVariableOpCAdam/sequential_3/batch_normalization_3/gamma/v/Read/ReadVariableOpBAdam/sequential_3/batch_normalization_3/beta/v/Read/ReadVariableOp7Adam/sequential_4/conv2d_4/kernel/v/Read/ReadVariableOp5Adam/sequential_4/conv2d_4/bias/v/Read/ReadVariableOpCAdam/sequential_4/batch_normalization_4/gamma/v/Read/ReadVariableOpBAdam/sequential_4/batch_normalization_4/beta/v/Read/ReadVariableOp7Adam/sequential_5/conv2d_5/kernel/v/Read/ReadVariableOp5Adam/sequential_5/conv2d_5/bias/v/Read/ReadVariableOpCAdam/sequential_5/batch_normalization_5/gamma/v/Read/ReadVariableOpBAdam/sequential_5/batch_normalization_5/beta/v/Read/ReadVariableOp4Adam/sequential_6/dense/kernel/v/Read/ReadVariableOp2Adam/sequential_6/dense/bias/v/Read/ReadVariableOpConst*p
Tini
g2e	*
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
__inference__traced_save_30907
ò
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratesequential/conv2d/kernelsequential/conv2d/bias$sequential/batch_normalization/gamma#sequential/batch_normalization/beta*sequential/batch_normalization/moving_mean.sequential/batch_normalization/moving_variancesequential_1/conv2d_1/kernelsequential_1/conv2d_1/bias(sequential_1/batch_normalization_1/gamma'sequential_1/batch_normalization_1/beta.sequential_1/batch_normalization_1/moving_mean2sequential_1/batch_normalization_1/moving_variancesequential_2/conv2d_2/kernelsequential_2/conv2d_2/bias(sequential_2/batch_normalization_2/gamma'sequential_2/batch_normalization_2/beta.sequential_2/batch_normalization_2/moving_mean2sequential_2/batch_normalization_2/moving_variancesequential_3/conv2d_3/kernelsequential_3/conv2d_3/bias(sequential_3/batch_normalization_3/gamma'sequential_3/batch_normalization_3/beta.sequential_3/batch_normalization_3/moving_mean2sequential_3/batch_normalization_3/moving_variancesequential_4/conv2d_4/kernelsequential_4/conv2d_4/bias(sequential_4/batch_normalization_4/gamma'sequential_4/batch_normalization_4/beta.sequential_4/batch_normalization_4/moving_mean2sequential_4/batch_normalization_4/moving_variancesequential_5/conv2d_5/kernelsequential_5/conv2d_5/bias(sequential_5/batch_normalization_5/gamma'sequential_5/batch_normalization_5/beta.sequential_5/batch_normalization_5/moving_mean2sequential_5/batch_normalization_5/moving_variancesequential_6/dense/kernelsequential_6/dense/biastotalcounttotal_1count_1Adam/sequential/conv2d/kernel/mAdam/sequential/conv2d/bias/m+Adam/sequential/batch_normalization/gamma/m*Adam/sequential/batch_normalization/beta/m#Adam/sequential_1/conv2d_1/kernel/m!Adam/sequential_1/conv2d_1/bias/m/Adam/sequential_1/batch_normalization_1/gamma/m.Adam/sequential_1/batch_normalization_1/beta/m#Adam/sequential_2/conv2d_2/kernel/m!Adam/sequential_2/conv2d_2/bias/m/Adam/sequential_2/batch_normalization_2/gamma/m.Adam/sequential_2/batch_normalization_2/beta/m#Adam/sequential_3/conv2d_3/kernel/m!Adam/sequential_3/conv2d_3/bias/m/Adam/sequential_3/batch_normalization_3/gamma/m.Adam/sequential_3/batch_normalization_3/beta/m#Adam/sequential_4/conv2d_4/kernel/m!Adam/sequential_4/conv2d_4/bias/m/Adam/sequential_4/batch_normalization_4/gamma/m.Adam/sequential_4/batch_normalization_4/beta/m#Adam/sequential_5/conv2d_5/kernel/m!Adam/sequential_5/conv2d_5/bias/m/Adam/sequential_5/batch_normalization_5/gamma/m.Adam/sequential_5/batch_normalization_5/beta/m Adam/sequential_6/dense/kernel/mAdam/sequential_6/dense/bias/mAdam/sequential/conv2d/kernel/vAdam/sequential/conv2d/bias/v+Adam/sequential/batch_normalization/gamma/v*Adam/sequential/batch_normalization/beta/v#Adam/sequential_1/conv2d_1/kernel/v!Adam/sequential_1/conv2d_1/bias/v/Adam/sequential_1/batch_normalization_1/gamma/v.Adam/sequential_1/batch_normalization_1/beta/v#Adam/sequential_2/conv2d_2/kernel/v!Adam/sequential_2/conv2d_2/bias/v/Adam/sequential_2/batch_normalization_2/gamma/v.Adam/sequential_2/batch_normalization_2/beta/v#Adam/sequential_3/conv2d_3/kernel/v!Adam/sequential_3/conv2d_3/bias/v/Adam/sequential_3/batch_normalization_3/gamma/v.Adam/sequential_3/batch_normalization_3/beta/v#Adam/sequential_4/conv2d_4/kernel/v!Adam/sequential_4/conv2d_4/bias/v/Adam/sequential_4/batch_normalization_4/gamma/v.Adam/sequential_4/batch_normalization_4/beta/v#Adam/sequential_5/conv2d_5/kernel/v!Adam/sequential_5/conv2d_5/bias/v/Adam/sequential_5/batch_normalization_5/gamma/v.Adam/sequential_5/batch_normalization_5/beta/v Adam/sequential_6/dense/kernel/vAdam/sequential_6/dense/bias/v*o
Tinh
f2d*
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
!__inference__traced_restore_31214¾£$
¤
Å
,__inference_sequential_2_layer_call_fn_26084
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
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
GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_260692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_2_input


P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25545

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
È
­
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26194

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
Æ
«
N__inference_batch_normalization_layer_call_and_return_conditional_losses_29598

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
®
«
C__inference_conv2d_1_layer_call_and_return_conditional_losses_29716

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

}
(__inference_conv2d_3_layer_call_fn_30065

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
C__inference_conv2d_3_layer_call_and_return_conditional_losses_262622
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

­
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_29938

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
È
­
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25514

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
ø

,__inference_sequential_6_layer_call_fn_29536

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
G__inference_sequential_6_layer_call_and_return_conditional_losses_272422
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
³
¸
,__inference_functional_1_layer_call_fn_28775

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

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity¢StatefulPartitionedCallé
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_279732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ê
_input_shapes¸
µ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

½
,__inference_sequential_1_layer_call_fn_28984

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
G__inference_sequential_1_layer_call_and_return_conditional_losses_257422
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
Ø
í
G__inference_sequential_1_layer_call_and_return_conditional_losses_25719
conv2d_1_input
conv2d_1_25702
conv2d_1_25704
batch_normalization_1_25708
batch_normalization_1_25710
batch_normalization_1_25712
batch_normalization_1_25714
identity¢-batch_normalization_1/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¦
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_25702conv2d_1_25704*
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_255822"
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
G__inference_activation_2_layer_call_and_return_conditional_losses_256162
activation_2/PartitionedCall¼
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0batch_normalization_1_25708batch_normalization_1_25710batch_normalization_1_25712batch_normalization_1_25714*
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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_256612/
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_255622
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

Á
*__inference_sequential_layer_call_fn_25440
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
E__inference_sequential_layer_call_and_return_conditional_losses_254252
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


,__inference_sequential_6_layer_call_fn_27249
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_272422
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
¤

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26567

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
Ô
­
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_26874

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

°
#__inference_signature_wrapper_28143
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

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity¢StatefulPartitionedCallÃ
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_251142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ê
_input_shapes¸
µ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
à
¨
5__inference_batch_normalization_1_layer_call_fn_29876

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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_256612
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


P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_30020

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
¥
c
G__inference_activation_5_layer_call_and_return_conditional_losses_30253

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

­
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_30512

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
¦
Å
,__inference_sequential_1_layer_call_fn_25794
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_257792
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
÷
r
F__inference_concatenate_layer_call_and_return_conditional_losses_29008
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
å&
³
G__inference_sequential_1_layer_call_and_return_conditional_losses_28967

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
¬
«
C__inference_conv2d_5_layer_call_and_return_conditional_losses_26942

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

½
,__inference_sequential_4_layer_call_fn_29348

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
:ÿÿÿÿÿÿÿÿÿ@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_267492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs

Ù
E__inference_sequential_layer_call_and_return_conditional_losses_25367
conv2d_input
conv2d_25351
conv2d_25353
batch_normalization_25357
batch_normalization_25359
batch_normalization_25361
batch_normalization_25363
identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_25351conv2d_25353*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_252322 
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
G__inference_activation_1_layer_call_and_return_conditional_losses_252662
activation_1/PartitionedCall®
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_25357batch_normalization_25359batch_normalization_25361batch_normalization_25363*
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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_253112-
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
Ô
­
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26536

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
Ã
H
,__inference_activation_2_layer_call_fn_29748

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
G__inference_activation_2_layer_call_and_return_conditional_losses_256162
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
È
­
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_30108

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


N__inference_batch_normalization_layer_call_and_return_conditional_losses_25207

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

}
(__inference_conv2d_2_layer_call_fn_29895

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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_259122
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

­
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26653

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
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_26806

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
É

¯
G__inference_sequential_6_layer_call_and_return_conditional_losses_27210
flatten_input
dense_27204
dense_27206
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
B__inference_flatten_layer_call_and_return_conditional_losses_271642
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_27204dense_27206*
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
@__inference_dense_layer_call_and_return_conditional_losses_271832
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
´E
ù	
G__inference_functional_1_layer_call_and_return_conditional_losses_27973

inputs
sequential_27883
sequential_27885
sequential_27887
sequential_27889
sequential_27891
sequential_27893
sequential_1_27897
sequential_1_27899
sequential_1_27901
sequential_1_27903
sequential_1_27905
sequential_1_27907
sequential_2_27911
sequential_2_27913
sequential_2_27915
sequential_2_27917
sequential_2_27919
sequential_2_27921
sequential_3_27925
sequential_3_27927
sequential_3_27929
sequential_3_27931
sequential_3_27933
sequential_3_27935
sequential_4_27939
sequential_4_27941
sequential_4_27943
sequential_4_27945
sequential_4_27947
sequential_4_27949
sequential_5_27953
sequential_5_27955
sequential_5_27957
sequential_5_27959
sequential_5_27961
sequential_5_27963
sequential_6_27967
sequential_6_27969
identity¢"sequential/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCall¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall¢$sequential_4/StatefulPartitionedCall¢$sequential_5/StatefulPartitionedCall¢$sequential_6/StatefulPartitionedCallø
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_27883sequential_27885sequential_27887sequential_27889sequential_27891sequential_27893*
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
E__inference_sequential_layer_call_and_return_conditional_losses_254252$
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
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_254462!
max_pooling2d_3/PartitionedCall¯
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_27897sequential_1_27899sequential_1_27901sequential_1_27903sequential_1_27905sequential_1_27907*
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_257792&
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
F__inference_concatenate_layer_call_and_return_conditional_losses_273552
concatenate/PartitionedCall¨
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sequential_2_27911sequential_2_27913sequential_2_27915sequential_2_27917sequential_2_27919sequential_2_27921*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_261052&
$sequential_2/StatefulPartitionedCall
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
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_261262!
max_pooling2d_4/PartitionedCall¯
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_27925sequential_3_27927sequential_3_27929sequential_3_27931sequential_3_27933sequential_3_27935*
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
G__inference_sequential_3_layer_call_and_return_conditional_losses_264592&
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
H__inference_concatenate_1_layer_call_and_return_conditional_losses_274662
concatenate_1/PartitionedCall©
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0sequential_4_27939sequential_4_27941sequential_4_27943sequential_4_27945sequential_4_27947sequential_4_27949*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_267852&
$sequential_4/StatefulPartitionedCall
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
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_268062!
max_pooling2d_5/PartitionedCall°
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0sequential_5_27953sequential_5_27955sequential_5_27957sequential_5_27959sequential_5_27961sequential_5_27963*
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
G__inference_sequential_5_layer_call_and_return_conditional_losses_271392&
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
H__inference_concatenate_2_layer_call_and_return_conditional_losses_275772
concatenate_2/PartitionedCallÈ
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0sequential_6_27967sequential_6_27969*
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_272422&
$sequential_6/StatefulPartitionedCall
IdentityIdentity-sequential_6/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ê
_input_shapes¸
µ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_25991

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
 
¨
5__inference_batch_normalization_2_layer_call_fn_30046

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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_258872
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

Á
*__inference_sequential_layer_call_fn_25404
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
E__inference_sequential_layer_call_and_return_conditional_losses_253892
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

­
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26323

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
Ç$
¥
E__inference_sequential_layer_call_and_return_conditional_losses_28853

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

»
*__inference_sequential_layer_call_fn_28870

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
E__inference_sequential_layer_call_and_return_conditional_losses_253892
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

{
&__inference_conv2d_layer_call_fn_29555

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
A__inference_conv2d_layer_call_and_return_conditional_losses_252322
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


P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26225

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
®
K
/__inference_max_pooling2d_1_layer_call_fn_26248

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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_262422
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

}
(__inference_conv2d_1_layer_call_fn_29725

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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_255822
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
ª
I
-__inference_max_pooling2d_layer_call_fn_25568

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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_255622
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
¢
Å
,__inference_sequential_5_layer_call_fn_27154
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
G__inference_sequential_5_layer_call_and_return_conditional_losses_271392
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
Ã
H
,__inference_activation_1_layer_call_fn_29578

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
G__inference_activation_1_layer_call_and_return_conditional_losses_252662
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
Û

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_30360

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
¿
H
,__inference_activation_5_layer_call_fn_30258

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
G__inference_activation_5_layer_call_and_return_conditional_losses_266262
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
®
å
G__inference_sequential_2_layer_call_and_return_conditional_losses_26105

inputs
conv2d_2_26089
conv2d_2_26091
batch_normalization_2_26095
batch_normalization_2_26097
batch_normalization_2_26099
batch_normalization_2_26101
identity¢-batch_normalization_2/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_26089conv2d_2_26091*
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_259122"
 conv2d_2/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
G__inference_activation_3_layer_call_and_return_conditional_losses_259462
activation_3/PartitionedCall¼
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0batch_normalization_2_26095batch_normalization_2_26097batch_normalization_2_26099batch_normalization_2_26101*
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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_259912/
-batch_normalization_2/StatefulPartitionedCallç
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_26905

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
¬
©
A__inference_conv2d_layer_call_and_return_conditional_losses_25232

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

¨
5__inference_batch_normalization_1_layer_call_fn_29799

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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_255142
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
Ü
¨
5__inference_batch_normalization_5_layer_call_fn_30556

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
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_270212
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
§
¸
,__inference_functional_1_layer_call_fn_28694

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

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity¢StatefulPartitionedCallÝ
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !"%&*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_277992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ê
_input_shapes¸
µ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
å
G__inference_sequential_4_layer_call_and_return_conditional_losses_26749

inputs
conv2d_4_26733
conv2d_4_26735
batch_normalization_4_26739
batch_normalization_4_26741
batch_normalization_4_26743
batch_normalization_4_26745
identity¢-batch_normalization_4/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_26733conv2d_4_26735*
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_265922"
 conv2d_4/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
G__inference_activation_5_layer_call_and_return_conditional_losses_266262
activation_5/PartitionedCall¹
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0batch_normalization_4_26739batch_normalization_4_26741batch_normalization_4_26743batch_normalization_4_26745*
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
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_266532/
-batch_normalization_4/StatefulPartitionedCallæ
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ô
Ó
E__inference_sequential_layer_call_and_return_conditional_losses_25425

inputs
conv2d_25409
conv2d_25411
batch_normalization_25415
batch_normalization_25417
batch_normalization_25419
batch_normalization_25421
identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_25409conv2d_25411*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_252322 
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
G__inference_activation_1_layer_call_and_return_conditional_losses_252662
activation_1/PartitionedCall®
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_25415batch_normalization_25417batch_normalization_25419batch_normalization_25421*
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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_253112-
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
à
¨
5__inference_batch_normalization_2_layer_call_fn_29982

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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_259912
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
Þ
¨
5__inference_batch_normalization_2_layer_call_fn_29969

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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_259732
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
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27021

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
×

P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_29956

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
Ô
­
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_30448

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
ò
Ó
E__inference_sequential_layer_call_and_return_conditional_losses_25389

inputs
conv2d_25373
conv2d_25375
batch_normalization_25379
batch_normalization_25381
batch_normalization_25383
batch_normalization_25385
identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_25373conv2d_25375*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_252322 
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
G__inference_activation_1_layer_call_and_return_conditional_losses_252662
activation_1/PartitionedCall¬
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_25379batch_normalization_25381batch_normalization_25383batch_normalization_25385*
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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_252932-
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

­
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25643

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
Ü
í
G__inference_sequential_3_layer_call_and_return_conditional_losses_26399
conv2d_3_input
conv2d_3_26382
conv2d_3_26384
batch_normalization_3_26388
batch_normalization_3_26390
batch_normalization_3_26392
batch_normalization_3_26394
identity¢-batch_normalization_3/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¦
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_3_26382conv2d_3_26384*
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
C__inference_conv2d_3_layer_call_and_return_conditional_losses_262622"
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
G__inference_activation_4_layer_call_and_return_conditional_losses_262962
activation_4/PartitionedCall¼
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0batch_normalization_3_26388batch_normalization_3_26390batch_normalization_3_26392batch_normalization_3_26394*
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
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_263412/
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_262422!
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
 
Å
,__inference_sequential_4_layer_call_fn_26764
conv2d_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
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
GPU2*0J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_267492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
(
_user_specified_nameconv2d_4_input
Æ
«
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25176

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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_25661

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
×

P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29850

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
Þ
¨
5__inference_batch_normalization_1_layer_call_fn_29863

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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_256432
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
Ä
å
G__inference_sequential_3_layer_call_and_return_conditional_losses_26459

inputs
conv2d_3_26442
conv2d_3_26444
batch_normalization_3_26448
batch_normalization_3_26450
batch_normalization_3_26452
batch_normalization_3_26454
identity¢-batch_normalization_3/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_26442conv2d_3_26444*
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
C__inference_conv2d_3_layer_call_and_return_conditional_losses_262622"
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
G__inference_activation_4_layer_call_and_return_conditional_losses_262962
activation_4/PartitionedCall¼
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0batch_normalization_3_26448batch_normalization_3_26450batch_normalization_3_26452batch_normalization_3_26454*
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
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_263412/
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_262422!
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

­
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27003

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
®
«
C__inference_conv2d_2_layer_call_and_return_conditional_losses_29886

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

½
,__inference_sequential_3_layer_call_fn_29223

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
G__inference_sequential_3_layer_call_and_return_conditional_losses_264222
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

­
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_30342

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
¨
å
G__inference_sequential_4_layer_call_and_return_conditional_losses_26785

inputs
conv2d_4_26769
conv2d_4_26771
batch_normalization_4_26775
batch_normalization_4_26777
batch_normalization_4_26779
batch_normalization_4_26781
identity¢-batch_normalization_4/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_26769conv2d_4_26771*
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_265922"
 conv2d_4/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
G__inference_activation_5_layer_call_and_return_conditional_losses_266262
activation_5/PartitionedCall»
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0batch_normalization_4_26775batch_normalization_4_26777batch_normalization_4_26779batch_normalization_4_26781*
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
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_266712/
-batch_normalization_4/StatefulPartitionedCallæ
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
.

G__inference_sequential_4_layer_call_and_return_conditional_losses_29293

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_4/AssignNewValue¢&batch_normalization_4/AssignNewValue_1²
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp¿
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
2
conv2d_4/Conv2D¨
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
conv2d_4/BiasAddm
activation_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_5/mul/x
activation_5/mulMulactivation_5/mul/x:output:0conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/mulo
activation_5/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
activation_5/Sqrt/xm
activation_5/SqrtSqrtactivation_5/Sqrt/x:output:0*
T0*
_output_shapes
: 2
activation_5/Sqrtm
activation_5/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
activation_5/Pow/y
activation_5/PowPowconv2d_4/BiasAdd:output:0activation_5/Pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/Powq
activation_5/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2
activation_5/mul_1/x
activation_5/mul_1Mulactivation_5/mul_1/x:output:0activation_5/Pow:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/mul_1
activation_5/addAddV2conv2d_4/BiasAdd:output:0activation_5/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/add
activation_5/mul_2Mulactivation_5/Sqrt:y:0activation_5/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/mul_2
activation_5/TanhTanhactivation_5/mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/Tanhq
activation_5/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_5/add_1/x¢
activation_5/add_1AddV2activation_5/add_1/x:output:0activation_5/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/add_1
activation_5/mul_3Mulactivation_5/mul:z:0activation_5/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/mul_3·
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_4/ReadVariableOp½
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ê
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ñ
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_5/mul_3:z:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_4/FusedBatchNormV3
$batch_normalization_4/AssignNewValueAssignVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValue
&batch_normalization_4/AssignNewValue_1AssignVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1×
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::2L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
¤

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_30296

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
/__inference_max_pooling2d_4_layer_call_fn_26132

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
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_261262
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
¦
Å
,__inference_sequential_2_layer_call_fn_26120
conv2d_2_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_261052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_2_input
Â
»
G__inference_sequential_6_layer_call_and_return_conditional_losses_29518

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

­
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_25973

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
¬
«
C__inference_conv2d_4_layer_call_and_return_conditional_losses_30226

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
Û
z
%__inference_dense_layer_call_fn_30587

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
@__inference_dense_layer_call_and_return_conditional_losses_271832
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
³
¨
@__inference_dense_layer_call_and_return_conditional_losses_30578

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
¤
¨
5__inference_batch_normalization_5_layer_call_fn_30492

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
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_269052
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
Õ

N__inference_batch_normalization_layer_call_and_return_conditional_losses_29680

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
%
³
G__inference_sequential_2_layer_call_and_return_conditional_losses_29092

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource
identity°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpÀ
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp®
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_2/BiasAddm
activation_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_3/mul/x
activation_3/mulMulactivation_3/mul/x:output:0conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/mulo
activation_3/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
activation_3/Sqrt/xm
activation_3/SqrtSqrtactivation_3/Sqrt/x:output:0*
T0*
_output_shapes
: 2
activation_3/Sqrtm
activation_3/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
activation_3/Pow/y
activation_3/PowPowconv2d_2/BiasAdd:output:0activation_3/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/Powq
activation_3/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2
activation_3/mul_1/x 
activation_3/mul_1Mulactivation_3/mul_1/x:output:0activation_3/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/mul_1
activation_3/addAddV2conv2d_2/BiasAdd:output:0activation_3/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/add
activation_3/mul_2Mulactivation_3/Sqrt:y:0activation_3/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/mul_2
activation_3/TanhTanhactivation_3/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/Tanhq
activation_3/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_3/add_1/x£
activation_3/add_1AddV2activation_3/add_1/x:output:0activation_3/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/add_1
activation_3/mul_3Mulactivation_3/mul:z:0activation_3/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/mul_3¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1à
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_3/mul_3:z:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
IdentityIdentity*batch_normalization_2/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@:::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
°
c
G__inference_activation_1_layer_call_and_return_conditional_losses_29573

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
àß
Ó
G__inference_functional_1_layer_call_and_return_conditional_losses_28384

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
4sequential_2_conv2d_2_conv2d_readvariableop_resource9
5sequential_2_conv2d_2_biasadd_readvariableop_resource>
:sequential_2_batch_normalization_2_readvariableop_resource@
<sequential_2_batch_normalization_2_readvariableop_1_resourceO
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceQ
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource8
4sequential_3_conv2d_3_conv2d_readvariableop_resource9
5sequential_3_conv2d_3_biasadd_readvariableop_resource>
:sequential_3_batch_normalization_3_readvariableop_resource@
<sequential_3_batch_normalization_3_readvariableop_1_resourceO
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceQ
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource8
4sequential_4_conv2d_4_conv2d_readvariableop_resource9
5sequential_4_conv2d_4_biasadd_readvariableop_resource>
:sequential_4_batch_normalization_4_readvariableop_resource@
<sequential_4_batch_normalization_4_readvariableop_1_resourceO
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceQ
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource8
4sequential_5_conv2d_5_conv2d_readvariableop_resource9
5sequential_5_conv2d_5_biasadd_readvariableop_resource>
:sequential_5_batch_normalization_5_readvariableop_resource@
<sequential_5_batch_normalization_5_readvariableop_1_resourceO
Ksequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceQ
Msequential_5_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource5
1sequential_6_dense_matmul_readvariableop_resource6
2sequential_6_dense_biasadd_readvariableop_resource
identity¢-sequential/batch_normalization/AssignNewValue¢/sequential/batch_normalization/AssignNewValue_1¢1sequential_1/batch_normalization_1/AssignNewValue¢3sequential_1/batch_normalization_1/AssignNewValue_1¢1sequential_2/batch_normalization_2/AssignNewValue¢3sequential_2/batch_normalization_2/AssignNewValue_1¢1sequential_3/batch_normalization_3/AssignNewValue¢3sequential_3/batch_normalization_3/AssignNewValue_1¢1sequential_4/batch_normalization_4/AssignNewValue¢3sequential_4/batch_normalization_4/AssignNewValue_1¢1sequential_5/batch_normalization_5/AssignNewValue¢3sequential_5/batch_normalization_5/AssignNewValue_1Ë
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
concatenate/concat×
+sequential_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+sequential_2/conv2d_2/Conv2D/ReadVariableOpü
sequential_2/conv2d_2/Conv2DConv2Dconcatenate/concat:output:03sequential_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
sequential_2/conv2d_2/Conv2DÎ
,sequential_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_2/conv2d_2/BiasAdd/ReadVariableOpâ
sequential_2/conv2d_2/BiasAddBiasAdd%sequential_2/conv2d_2/Conv2D:output:04sequential_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_2/conv2d_2/BiasAdd
sequential_2/activation_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential_2/activation_3/mul/xÓ
sequential_2/activation_3/mulMul(sequential_2/activation_3/mul/x:output:0&sequential_2/conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_2/activation_3/mul
 sequential_2/activation_3/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2"
 sequential_2/activation_3/Sqrt/x
sequential_2/activation_3/SqrtSqrt)sequential_2/activation_3/Sqrt/x:output:0*
T0*
_output_shapes
: 2 
sequential_2/activation_3/Sqrt
sequential_2/activation_3/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2!
sequential_2/activation_3/Pow/yÓ
sequential_2/activation_3/PowPow&sequential_2/conv2d_2/BiasAdd:output:0(sequential_2/activation_3/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_2/activation_3/Pow
!sequential_2/activation_3/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2#
!sequential_2/activation_3/mul_1/xÔ
sequential_2/activation_3/mul_1Mul*sequential_2/activation_3/mul_1/x:output:0!sequential_2/activation_3/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_2/activation_3/mul_1Ð
sequential_2/activation_3/addAddV2&sequential_2/conv2d_2/BiasAdd:output:0#sequential_2/activation_3/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_2/activation_3/addÌ
sequential_2/activation_3/mul_2Mul"sequential_2/activation_3/Sqrt:y:0!sequential_2/activation_3/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_2/activation_3/mul_2©
sequential_2/activation_3/TanhTanh#sequential_2/activation_3/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
sequential_2/activation_3/Tanh
!sequential_2/activation_3/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!sequential_2/activation_3/add_1/x×
sequential_2/activation_3/add_1AddV2*sequential_2/activation_3/add_1/x:output:0"sequential_2/activation_3/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_2/activation_3/add_1Í
sequential_2/activation_3/mul_3Mul!sequential_2/activation_3/mul:z:0#sequential_2/activation_3/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_2/activation_3/mul_3Ý
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential_2/batch_normalization_2/ReadVariableOpã
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3sequential_2/batch_normalization_2/ReadVariableOp_1
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1É
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3#sequential_2/activation_3/mul_3:z:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<25
3sequential_2/batch_normalization_2/FusedBatchNormV3Ñ
1sequential_2/batch_normalization_2/AssignNewValueAssignVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource@sequential_2/batch_normalization_2/FusedBatchNormV3:batch_mean:0C^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1sequential_2/batch_normalization_2/AssignNewValueß
3sequential_2/batch_normalization_2/AssignNewValue_1AssignVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceDsequential_2/batch_normalization_2/FusedBatchNormV3:batch_variance:0E^sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3sequential_2/batch_normalization_2/AssignNewValue_1Ç
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
+sequential_3/conv2d_3/Conv2D/ReadVariableOp
sequential_3/conv2d_3/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
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
concatenate_1/concatÙ
+sequential_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_4_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+sequential_4/conv2d_4/Conv2D/ReadVariableOpý
sequential_4/conv2d_4/Conv2DConv2Dconcatenate_1/concat:output:03sequential_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
2
sequential_4/conv2d_4/Conv2DÏ
,sequential_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,sequential_4/conv2d_4/BiasAdd/ReadVariableOpá
sequential_4/conv2d_4/BiasAddBiasAdd%sequential_4/conv2d_4/Conv2D:output:04sequential_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_4/conv2d_4/BiasAdd
sequential_4/activation_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential_4/activation_5/mul/xÒ
sequential_4/activation_5/mulMul(sequential_4/activation_5/mul/x:output:0&sequential_4/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_4/activation_5/mul
 sequential_4/activation_5/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2"
 sequential_4/activation_5/Sqrt/x
sequential_4/activation_5/SqrtSqrt)sequential_4/activation_5/Sqrt/x:output:0*
T0*
_output_shapes
: 2 
sequential_4/activation_5/Sqrt
sequential_4/activation_5/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2!
sequential_4/activation_5/Pow/yÒ
sequential_4/activation_5/PowPow&sequential_4/conv2d_4/BiasAdd:output:0(sequential_4/activation_5/Pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_4/activation_5/Pow
!sequential_4/activation_5/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2#
!sequential_4/activation_5/mul_1/xÓ
sequential_4/activation_5/mul_1Mul*sequential_4/activation_5/mul_1/x:output:0!sequential_4/activation_5/Pow:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_4/activation_5/mul_1Ï
sequential_4/activation_5/addAddV2&sequential_4/conv2d_4/BiasAdd:output:0#sequential_4/activation_5/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_4/activation_5/addË
sequential_4/activation_5/mul_2Mul"sequential_4/activation_5/Sqrt:y:0!sequential_4/activation_5/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_4/activation_5/mul_2¨
sequential_4/activation_5/TanhTanh#sequential_4/activation_5/mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2 
sequential_4/activation_5/Tanh
!sequential_4/activation_5/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!sequential_4/activation_5/add_1/xÖ
sequential_4/activation_5/add_1AddV2*sequential_4/activation_5/add_1/x:output:0"sequential_4/activation_5/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_4/activation_5/add_1Ì
sequential_4/activation_5/mul_3Mul!sequential_4/activation_5/mul:z:0#sequential_4/activation_5/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_4/activation_5/mul_3Þ
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpä
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Ì
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3#sequential_4/activation_5/mul_3:z:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
exponential_avg_factor%
×#<25
3sequential_4/batch_normalization_4/FusedBatchNormV3Ñ
1sequential_4/batch_normalization_4/AssignNewValueAssignVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource@sequential_4/batch_normalization_4/FusedBatchNormV3:batch_mean:0C^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp*^
_classT
RPloc:@sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype023
1sequential_4/batch_normalization_4/AssignNewValueß
3sequential_4/batch_normalization_4/AssignNewValue_1AssignVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceDsequential_4/batch_normalization_4/FusedBatchNormV3:batch_variance:0E^sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*`
_classV
TRloc:@sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype025
3sequential_4/batch_normalization_4/AssignNewValue_1Ê
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
+sequential_5/conv2d_5/Conv2D/ReadVariableOp
sequential_5/conv2d_5/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:03sequential_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
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
sequential_6/dense/Softmaxì
IdentityIdentity$sequential_6/dense/Softmax:softmax:0.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_12^sequential_1/batch_normalization_1/AssignNewValue4^sequential_1/batch_normalization_1/AssignNewValue_12^sequential_2/batch_normalization_2/AssignNewValue4^sequential_2/batch_normalization_2/AssignNewValue_12^sequential_3/batch_normalization_3/AssignNewValue4^sequential_3/batch_normalization_3/AssignNewValue_12^sequential_4/batch_normalization_4/AssignNewValue4^sequential_4/batch_normalization_4/AssignNewValue_12^sequential_5/batch_normalization_5/AssignNewValue4^sequential_5/batch_normalization_5/AssignNewValue_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ê
_input_shapes¸
µ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12f
1sequential_1/batch_normalization_1/AssignNewValue1sequential_1/batch_normalization_1/AssignNewValue2j
3sequential_1/batch_normalization_1/AssignNewValue_13sequential_1/batch_normalization_1/AssignNewValue_12f
1sequential_2/batch_normalization_2/AssignNewValue1sequential_2/batch_normalization_2/AssignNewValue2j
3sequential_2/batch_normalization_2/AssignNewValue_13sequential_2/batch_normalization_2/AssignNewValue_12f
1sequential_3/batch_normalization_3/AssignNewValue1sequential_3/batch_normalization_3/AssignNewValue2j
3sequential_3/batch_normalization_3/AssignNewValue_13sequential_3/batch_normalization_3/AssignNewValue_12f
1sequential_4/batch_normalization_4/AssignNewValue1sequential_4/batch_normalization_4/AssignNewValue2j
3sequential_4/batch_normalization_4/AssignNewValue_13sequential_4/batch_normalization_4/AssignNewValue_12f
1sequential_5/batch_normalization_5/AssignNewValue1sequential_5/batch_normalization_5/AssignNewValue2j
3sequential_5/batch_normalization_5/AssignNewValue_13sequential_5/batch_normalization_5/AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
§
 __inference__wrapped_model_25114
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
Afunctional_1_sequential_2_conv2d_2_conv2d_readvariableop_resourceF
Bfunctional_1_sequential_2_conv2d_2_biasadd_readvariableop_resourceK
Gfunctional_1_sequential_2_batch_normalization_2_readvariableop_resourceM
Ifunctional_1_sequential_2_batch_normalization_2_readvariableop_1_resource\
Xfunctional_1_sequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource^
Zfunctional_1_sequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceE
Afunctional_1_sequential_3_conv2d_3_conv2d_readvariableop_resourceF
Bfunctional_1_sequential_3_conv2d_3_biasadd_readvariableop_resourceK
Gfunctional_1_sequential_3_batch_normalization_3_readvariableop_resourceM
Ifunctional_1_sequential_3_batch_normalization_3_readvariableop_1_resource\
Xfunctional_1_sequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource^
Zfunctional_1_sequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceE
Afunctional_1_sequential_4_conv2d_4_conv2d_readvariableop_resourceF
Bfunctional_1_sequential_4_conv2d_4_biasadd_readvariableop_resourceK
Gfunctional_1_sequential_4_batch_normalization_4_readvariableop_resourceM
Ifunctional_1_sequential_4_batch_normalization_4_readvariableop_1_resource\
Xfunctional_1_sequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource^
Zfunctional_1_sequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceE
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
functional_1/concatenate/concatþ
8functional_1/sequential_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOpAfunctional_1_sequential_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02:
8functional_1/sequential_2/conv2d_2/Conv2D/ReadVariableOp°
)functional_1/sequential_2/conv2d_2/Conv2DConv2D(functional_1/concatenate/concat:output:0@functional_1/sequential_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2+
)functional_1/sequential_2/conv2d_2/Conv2Dõ
9functional_1/sequential_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpBfunctional_1_sequential_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9functional_1/sequential_2/conv2d_2/BiasAdd/ReadVariableOp
*functional_1/sequential_2/conv2d_2/BiasAddBiasAdd2functional_1/sequential_2/conv2d_2/Conv2D:output:0Afunctional_1/sequential_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*functional_1/sequential_2/conv2d_2/BiasAdd¡
,functional_1/sequential_2/activation_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,functional_1/sequential_2/activation_3/mul/x
*functional_1/sequential_2/activation_3/mulMul5functional_1/sequential_2/activation_3/mul/x:output:03functional_1/sequential_2/conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*functional_1/sequential_2/activation_3/mul£
-functional_1/sequential_2/activation_3/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2/
-functional_1/sequential_2/activation_3/Sqrt/x»
+functional_1/sequential_2/activation_3/SqrtSqrt6functional_1/sequential_2/activation_3/Sqrt/x:output:0*
T0*
_output_shapes
: 2-
+functional_1/sequential_2/activation_3/Sqrt¡
,functional_1/sequential_2/activation_3/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2.
,functional_1/sequential_2/activation_3/Pow/y
*functional_1/sequential_2/activation_3/PowPow3functional_1/sequential_2/conv2d_2/BiasAdd:output:05functional_1/sequential_2/activation_3/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*functional_1/sequential_2/activation_3/Pow¥
.functional_1/sequential_2/activation_3/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=20
.functional_1/sequential_2/activation_3/mul_1/x
,functional_1/sequential_2/activation_3/mul_1Mul7functional_1/sequential_2/activation_3/mul_1/x:output:0.functional_1/sequential_2/activation_3/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,functional_1/sequential_2/activation_3/mul_1
*functional_1/sequential_2/activation_3/addAddV23functional_1/sequential_2/conv2d_2/BiasAdd:output:00functional_1/sequential_2/activation_3/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2,
*functional_1/sequential_2/activation_3/add
,functional_1/sequential_2/activation_3/mul_2Mul/functional_1/sequential_2/activation_3/Sqrt:y:0.functional_1/sequential_2/activation_3/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,functional_1/sequential_2/activation_3/mul_2Ð
+functional_1/sequential_2/activation_3/TanhTanh0functional_1/sequential_2/activation_3/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2-
+functional_1/sequential_2/activation_3/Tanh¥
.functional_1/sequential_2/activation_3/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.functional_1/sequential_2/activation_3/add_1/x
,functional_1/sequential_2/activation_3/add_1AddV27functional_1/sequential_2/activation_3/add_1/x:output:0/functional_1/sequential_2/activation_3/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,functional_1/sequential_2/activation_3/add_1
,functional_1/sequential_2/activation_3/mul_3Mul.functional_1/sequential_2/activation_3/mul:z:00functional_1/sequential_2/activation_3/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2.
,functional_1/sequential_2/activation_3/mul_3
>functional_1/sequential_2/batch_normalization_2/ReadVariableOpReadVariableOpGfunctional_1_sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02@
>functional_1/sequential_2/batch_normalization_2/ReadVariableOp
@functional_1/sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOpIfunctional_1_sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02B
@functional_1/sequential_2/batch_normalization_2/ReadVariableOp_1·
Ofunctional_1/sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpXfunctional_1_sequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02Q
Ofunctional_1/sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp½
Qfunctional_1/sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZfunctional_1_sequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02S
Qfunctional_1/sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1
@functional_1/sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV30functional_1/sequential_2/activation_3/mul_3:z:0Ffunctional_1/sequential_2/batch_normalization_2/ReadVariableOp:value:0Hfunctional_1/sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Wfunctional_1/sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Yfunctional_1/sequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2B
@functional_1/sequential_2/batch_normalization_2/FusedBatchNormV3î
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
8functional_1/sequential_3/conv2d_3/Conv2D/ReadVariableOpÌ
)functional_1/sequential_3/conv2d_3/Conv2DConv2DDfunctional_1/sequential_2/batch_normalization_2/FusedBatchNormV3:y:0@functional_1/sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
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
!functional_1/concatenate_1/concat
8functional_1/sequential_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOpAfunctional_1_sequential_4_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02:
8functional_1/sequential_4/conv2d_4/Conv2D/ReadVariableOp±
)functional_1/sequential_4/conv2d_4/Conv2DConv2D*functional_1/concatenate_1/concat:output:0@functional_1/sequential_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
2+
)functional_1/sequential_4/conv2d_4/Conv2Dö
9functional_1/sequential_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpBfunctional_1_sequential_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02;
9functional_1/sequential_4/conv2d_4/BiasAdd/ReadVariableOp
*functional_1/sequential_4/conv2d_4/BiasAddBiasAdd2functional_1/sequential_4/conv2d_4/Conv2D:output:0Afunctional_1/sequential_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2,
*functional_1/sequential_4/conv2d_4/BiasAdd¡
,functional_1/sequential_4/activation_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2.
,functional_1/sequential_4/activation_5/mul/x
*functional_1/sequential_4/activation_5/mulMul5functional_1/sequential_4/activation_5/mul/x:output:03functional_1/sequential_4/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2,
*functional_1/sequential_4/activation_5/mul£
-functional_1/sequential_4/activation_5/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2/
-functional_1/sequential_4/activation_5/Sqrt/x»
+functional_1/sequential_4/activation_5/SqrtSqrt6functional_1/sequential_4/activation_5/Sqrt/x:output:0*
T0*
_output_shapes
: 2-
+functional_1/sequential_4/activation_5/Sqrt¡
,functional_1/sequential_4/activation_5/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2.
,functional_1/sequential_4/activation_5/Pow/y
*functional_1/sequential_4/activation_5/PowPow3functional_1/sequential_4/conv2d_4/BiasAdd:output:05functional_1/sequential_4/activation_5/Pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2,
*functional_1/sequential_4/activation_5/Pow¥
.functional_1/sequential_4/activation_5/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=20
.functional_1/sequential_4/activation_5/mul_1/x
,functional_1/sequential_4/activation_5/mul_1Mul7functional_1/sequential_4/activation_5/mul_1/x:output:0.functional_1/sequential_4/activation_5/Pow:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2.
,functional_1/sequential_4/activation_5/mul_1
*functional_1/sequential_4/activation_5/addAddV23functional_1/sequential_4/conv2d_4/BiasAdd:output:00functional_1/sequential_4/activation_5/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2,
*functional_1/sequential_4/activation_5/addÿ
,functional_1/sequential_4/activation_5/mul_2Mul/functional_1/sequential_4/activation_5/Sqrt:y:0.functional_1/sequential_4/activation_5/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2.
,functional_1/sequential_4/activation_5/mul_2Ï
+functional_1/sequential_4/activation_5/TanhTanh0functional_1/sequential_4/activation_5/mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2-
+functional_1/sequential_4/activation_5/Tanh¥
.functional_1/sequential_4/activation_5/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.functional_1/sequential_4/activation_5/add_1/x
,functional_1/sequential_4/activation_5/add_1AddV27functional_1/sequential_4/activation_5/add_1/x:output:0/functional_1/sequential_4/activation_5/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2.
,functional_1/sequential_4/activation_5/add_1
,functional_1/sequential_4/activation_5/mul_3Mul.functional_1/sequential_4/activation_5/mul:z:00functional_1/sequential_4/activation_5/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2.
,functional_1/sequential_4/activation_5/mul_3
>functional_1/sequential_4/batch_normalization_4/ReadVariableOpReadVariableOpGfunctional_1_sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02@
>functional_1/sequential_4/batch_normalization_4/ReadVariableOp
@functional_1/sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOpIfunctional_1_sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02B
@functional_1/sequential_4/batch_normalization_4/ReadVariableOp_1¸
Ofunctional_1/sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpXfunctional_1_sequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02Q
Ofunctional_1/sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp¾
Qfunctional_1/sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpZfunctional_1_sequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02S
Qfunctional_1/sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1
@functional_1/sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV30functional_1/sequential_4/activation_5/mul_3:z:0Ffunctional_1/sequential_4/batch_normalization_4/ReadVariableOp:value:0Hfunctional_1/sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Wfunctional_1/sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Yfunctional_1/sequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
is_training( 2B
@functional_1/sequential_4/batch_normalization_4/FusedBatchNormV3ñ
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
8functional_1/sequential_5/conv2d_5/Conv2D/ReadVariableOpË
)functional_1/sequential_5/conv2d_5/Conv2DConv2DDfunctional_1/sequential_4/batch_normalization_4/FusedBatchNormV3:y:0@functional_1/sequential_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
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
identityIdentity:output:0*Ê
_input_shapes¸
µ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

½
,__inference_sequential_4_layer_call_fn_29365

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
:ÿÿÿÿÿÿÿÿÿ@@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_267852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs
ë
r
H__inference_concatenate_2_layer_call_and_return_conditional_losses_27577

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
ø

,__inference_sequential_6_layer_call_fn_29527

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
G__inference_sequential_6_layer_call_and_return_conditional_losses_272232
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
ï
p
F__inference_concatenate_layer_call_and_return_conditional_losses_27355

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
Ê
ß5
__inference__traced_save_30907
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop7
3savev2_sequential_conv2d_kernel_read_readvariableop5
1savev2_sequential_conv2d_bias_read_readvariableopC
?savev2_sequential_batch_normalization_gamma_read_readvariableopB
>savev2_sequential_batch_normalization_beta_read_readvariableopI
Esavev2_sequential_batch_normalization_moving_mean_read_readvariableopM
Isavev2_sequential_batch_normalization_moving_variance_read_readvariableop;
7savev2_sequential_1_conv2d_1_kernel_read_readvariableop9
5savev2_sequential_1_conv2d_1_bias_read_readvariableopG
Csavev2_sequential_1_batch_normalization_1_gamma_read_readvariableopF
Bsavev2_sequential_1_batch_normalization_1_beta_read_readvariableopM
Isavev2_sequential_1_batch_normalization_1_moving_mean_read_readvariableopQ
Msavev2_sequential_1_batch_normalization_1_moving_variance_read_readvariableop;
7savev2_sequential_2_conv2d_2_kernel_read_readvariableop9
5savev2_sequential_2_conv2d_2_bias_read_readvariableopG
Csavev2_sequential_2_batch_normalization_2_gamma_read_readvariableopF
Bsavev2_sequential_2_batch_normalization_2_beta_read_readvariableopM
Isavev2_sequential_2_batch_normalization_2_moving_mean_read_readvariableopQ
Msavev2_sequential_2_batch_normalization_2_moving_variance_read_readvariableop;
7savev2_sequential_3_conv2d_3_kernel_read_readvariableop9
5savev2_sequential_3_conv2d_3_bias_read_readvariableopG
Csavev2_sequential_3_batch_normalization_3_gamma_read_readvariableopF
Bsavev2_sequential_3_batch_normalization_3_beta_read_readvariableopM
Isavev2_sequential_3_batch_normalization_3_moving_mean_read_readvariableopQ
Msavev2_sequential_3_batch_normalization_3_moving_variance_read_readvariableop;
7savev2_sequential_4_conv2d_4_kernel_read_readvariableop9
5savev2_sequential_4_conv2d_4_bias_read_readvariableopG
Csavev2_sequential_4_batch_normalization_4_gamma_read_readvariableopF
Bsavev2_sequential_4_batch_normalization_4_beta_read_readvariableopM
Isavev2_sequential_4_batch_normalization_4_moving_mean_read_readvariableopQ
Msavev2_sequential_4_batch_normalization_4_moving_variance_read_readvariableop;
7savev2_sequential_5_conv2d_5_kernel_read_readvariableop9
5savev2_sequential_5_conv2d_5_bias_read_readvariableopG
Csavev2_sequential_5_batch_normalization_5_gamma_read_readvariableopF
Bsavev2_sequential_5_batch_normalization_5_beta_read_readvariableopM
Isavev2_sequential_5_batch_normalization_5_moving_mean_read_readvariableopQ
Msavev2_sequential_5_batch_normalization_5_moving_variance_read_readvariableop8
4savev2_sequential_6_dense_kernel_read_readvariableop6
2savev2_sequential_6_dense_bias_read_readvariableop$
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
>savev2_adam_sequential_2_conv2d_2_kernel_m_read_readvariableop@
<savev2_adam_sequential_2_conv2d_2_bias_m_read_readvariableopN
Jsavev2_adam_sequential_2_batch_normalization_2_gamma_m_read_readvariableopM
Isavev2_adam_sequential_2_batch_normalization_2_beta_m_read_readvariableopB
>savev2_adam_sequential_3_conv2d_3_kernel_m_read_readvariableop@
<savev2_adam_sequential_3_conv2d_3_bias_m_read_readvariableopN
Jsavev2_adam_sequential_3_batch_normalization_3_gamma_m_read_readvariableopM
Isavev2_adam_sequential_3_batch_normalization_3_beta_m_read_readvariableopB
>savev2_adam_sequential_4_conv2d_4_kernel_m_read_readvariableop@
<savev2_adam_sequential_4_conv2d_4_bias_m_read_readvariableopN
Jsavev2_adam_sequential_4_batch_normalization_4_gamma_m_read_readvariableopM
Isavev2_adam_sequential_4_batch_normalization_4_beta_m_read_readvariableopB
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
>savev2_adam_sequential_2_conv2d_2_kernel_v_read_readvariableop@
<savev2_adam_sequential_2_conv2d_2_bias_v_read_readvariableopN
Jsavev2_adam_sequential_2_batch_normalization_2_gamma_v_read_readvariableopM
Isavev2_adam_sequential_2_batch_normalization_2_beta_v_read_readvariableopB
>savev2_adam_sequential_3_conv2d_3_kernel_v_read_readvariableop@
<savev2_adam_sequential_3_conv2d_3_bias_v_read_readvariableopN
Jsavev2_adam_sequential_3_batch_normalization_3_gamma_v_read_readvariableopM
Isavev2_adam_sequential_3_batch_normalization_3_beta_v_read_readvariableopB
>savev2_adam_sequential_4_conv2d_4_kernel_v_read_readvariableop@
<savev2_adam_sequential_4_conv2d_4_bias_v_read_readvariableopN
Jsavev2_adam_sequential_4_batch_normalization_4_gamma_v_read_readvariableopM
Isavev2_adam_sequential_4_batch_normalization_4_beta_v_read_readvariableopB
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
value3B1 B+_temp_d28ef6ce36204e18a3ace02bf2aae31c/part2	
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
ShardedFilenameÞ,
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*ð+
valueæ+Bã+dB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÓ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ý
valueÓBÐdB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices4
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop3savev2_sequential_conv2d_kernel_read_readvariableop1savev2_sequential_conv2d_bias_read_readvariableop?savev2_sequential_batch_normalization_gamma_read_readvariableop>savev2_sequential_batch_normalization_beta_read_readvariableopEsavev2_sequential_batch_normalization_moving_mean_read_readvariableopIsavev2_sequential_batch_normalization_moving_variance_read_readvariableop7savev2_sequential_1_conv2d_1_kernel_read_readvariableop5savev2_sequential_1_conv2d_1_bias_read_readvariableopCsavev2_sequential_1_batch_normalization_1_gamma_read_readvariableopBsavev2_sequential_1_batch_normalization_1_beta_read_readvariableopIsavev2_sequential_1_batch_normalization_1_moving_mean_read_readvariableopMsavev2_sequential_1_batch_normalization_1_moving_variance_read_readvariableop7savev2_sequential_2_conv2d_2_kernel_read_readvariableop5savev2_sequential_2_conv2d_2_bias_read_readvariableopCsavev2_sequential_2_batch_normalization_2_gamma_read_readvariableopBsavev2_sequential_2_batch_normalization_2_beta_read_readvariableopIsavev2_sequential_2_batch_normalization_2_moving_mean_read_readvariableopMsavev2_sequential_2_batch_normalization_2_moving_variance_read_readvariableop7savev2_sequential_3_conv2d_3_kernel_read_readvariableop5savev2_sequential_3_conv2d_3_bias_read_readvariableopCsavev2_sequential_3_batch_normalization_3_gamma_read_readvariableopBsavev2_sequential_3_batch_normalization_3_beta_read_readvariableopIsavev2_sequential_3_batch_normalization_3_moving_mean_read_readvariableopMsavev2_sequential_3_batch_normalization_3_moving_variance_read_readvariableop7savev2_sequential_4_conv2d_4_kernel_read_readvariableop5savev2_sequential_4_conv2d_4_bias_read_readvariableopCsavev2_sequential_4_batch_normalization_4_gamma_read_readvariableopBsavev2_sequential_4_batch_normalization_4_beta_read_readvariableopIsavev2_sequential_4_batch_normalization_4_moving_mean_read_readvariableopMsavev2_sequential_4_batch_normalization_4_moving_variance_read_readvariableop7savev2_sequential_5_conv2d_5_kernel_read_readvariableop5savev2_sequential_5_conv2d_5_bias_read_readvariableopCsavev2_sequential_5_batch_normalization_5_gamma_read_readvariableopBsavev2_sequential_5_batch_normalization_5_beta_read_readvariableopIsavev2_sequential_5_batch_normalization_5_moving_mean_read_readvariableopMsavev2_sequential_5_batch_normalization_5_moving_variance_read_readvariableop4savev2_sequential_6_dense_kernel_read_readvariableop2savev2_sequential_6_dense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop:savev2_adam_sequential_conv2d_kernel_m_read_readvariableop8savev2_adam_sequential_conv2d_bias_m_read_readvariableopFsavev2_adam_sequential_batch_normalization_gamma_m_read_readvariableopEsavev2_adam_sequential_batch_normalization_beta_m_read_readvariableop>savev2_adam_sequential_1_conv2d_1_kernel_m_read_readvariableop<savev2_adam_sequential_1_conv2d_1_bias_m_read_readvariableopJsavev2_adam_sequential_1_batch_normalization_1_gamma_m_read_readvariableopIsavev2_adam_sequential_1_batch_normalization_1_beta_m_read_readvariableop>savev2_adam_sequential_2_conv2d_2_kernel_m_read_readvariableop<savev2_adam_sequential_2_conv2d_2_bias_m_read_readvariableopJsavev2_adam_sequential_2_batch_normalization_2_gamma_m_read_readvariableopIsavev2_adam_sequential_2_batch_normalization_2_beta_m_read_readvariableop>savev2_adam_sequential_3_conv2d_3_kernel_m_read_readvariableop<savev2_adam_sequential_3_conv2d_3_bias_m_read_readvariableopJsavev2_adam_sequential_3_batch_normalization_3_gamma_m_read_readvariableopIsavev2_adam_sequential_3_batch_normalization_3_beta_m_read_readvariableop>savev2_adam_sequential_4_conv2d_4_kernel_m_read_readvariableop<savev2_adam_sequential_4_conv2d_4_bias_m_read_readvariableopJsavev2_adam_sequential_4_batch_normalization_4_gamma_m_read_readvariableopIsavev2_adam_sequential_4_batch_normalization_4_beta_m_read_readvariableop>savev2_adam_sequential_5_conv2d_5_kernel_m_read_readvariableop<savev2_adam_sequential_5_conv2d_5_bias_m_read_readvariableopJsavev2_adam_sequential_5_batch_normalization_5_gamma_m_read_readvariableopIsavev2_adam_sequential_5_batch_normalization_5_beta_m_read_readvariableop;savev2_adam_sequential_6_dense_kernel_m_read_readvariableop9savev2_adam_sequential_6_dense_bias_m_read_readvariableop:savev2_adam_sequential_conv2d_kernel_v_read_readvariableop8savev2_adam_sequential_conv2d_bias_v_read_readvariableopFsavev2_adam_sequential_batch_normalization_gamma_v_read_readvariableopEsavev2_adam_sequential_batch_normalization_beta_v_read_readvariableop>savev2_adam_sequential_1_conv2d_1_kernel_v_read_readvariableop<savev2_adam_sequential_1_conv2d_1_bias_v_read_readvariableopJsavev2_adam_sequential_1_batch_normalization_1_gamma_v_read_readvariableopIsavev2_adam_sequential_1_batch_normalization_1_beta_v_read_readvariableop>savev2_adam_sequential_2_conv2d_2_kernel_v_read_readvariableop<savev2_adam_sequential_2_conv2d_2_bias_v_read_readvariableopJsavev2_adam_sequential_2_batch_normalization_2_gamma_v_read_readvariableopIsavev2_adam_sequential_2_batch_normalization_2_beta_v_read_readvariableop>savev2_adam_sequential_3_conv2d_3_kernel_v_read_readvariableop<savev2_adam_sequential_3_conv2d_3_bias_v_read_readvariableopJsavev2_adam_sequential_3_batch_normalization_3_gamma_v_read_readvariableopIsavev2_adam_sequential_3_batch_normalization_3_beta_v_read_readvariableop>savev2_adam_sequential_4_conv2d_4_kernel_v_read_readvariableop<savev2_adam_sequential_4_conv2d_4_bias_v_read_readvariableopJsavev2_adam_sequential_4_batch_normalization_4_gamma_v_read_readvariableopIsavev2_adam_sequential_4_batch_normalization_4_beta_v_read_readvariableop>savev2_adam_sequential_5_conv2d_5_kernel_v_read_readvariableop<savev2_adam_sequential_5_conv2d_5_bias_v_read_readvariableopJsavev2_adam_sequential_5_batch_normalization_5_gamma_v_read_readvariableopIsavev2_adam_sequential_5_batch_normalization_5_beta_v_read_readvariableop;savev2_adam_sequential_6_dense_kernel_v_read_readvariableop9savev2_adam_sequential_6_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *r
dtypesh
f2d	2
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

identity_1Identity_1:output:0*Ó
_input_shapesÁ
¾: : : : : : : : : : : : :  : : : : : :@@:@:@:@:@:@:@@:@:@:@:@:@:::::::::::::
:: : : : : : : : :  : : : :@@:@:@:@:@@:@:@:@:::::::::
:: : : : :  : : : :@@:@:@:@:@@:@:@:@:::::::::
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
: : 


_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:.*
(
_output_shapes
::!

_output_shapes	
::! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::!#

_output_shapes	
::.$*
(
_output_shapes
::!%

_output_shapes	
::!&

_output_shapes	
::!'

_output_shapes	
::!(

_output_shapes	
::!)

_output_shapes	
::&*"
 
_output_shapes
:
: +

_output_shapes
::,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :,0(
&
_output_shapes
: : 1

_output_shapes
: : 2

_output_shapes
: : 3

_output_shapes
: :,4(
&
_output_shapes
:  : 5

_output_shapes
: : 6

_output_shapes
: : 7

_output_shapes
: :,8(
&
_output_shapes
:@@: 9

_output_shapes
:@: :

_output_shapes
:@: ;

_output_shapes
:@:,<(
&
_output_shapes
:@@: =

_output_shapes
:@: >

_output_shapes
:@: ?

_output_shapes
:@:.@*
(
_output_shapes
::!A

_output_shapes	
::!B

_output_shapes	
::!C

_output_shapes	
::.D*
(
_output_shapes
::!E

_output_shapes	
::!F

_output_shapes	
::!G

_output_shapes	
::&H"
 
_output_shapes
:
: I

_output_shapes
::,J(
&
_output_shapes
: : K

_output_shapes
: : L

_output_shapes
: : M

_output_shapes
: :,N(
&
_output_shapes
:  : O

_output_shapes
: : P

_output_shapes
: : Q

_output_shapes
: :,R(
&
_output_shapes
:@@: S

_output_shapes
:@: T

_output_shapes
:@: U

_output_shapes
:@:,V(
&
_output_shapes
:@@: W

_output_shapes
:@: X

_output_shapes
:@: Y

_output_shapes
:@:.Z*
(
_output_shapes
::![

_output_shapes	
::!\

_output_shapes	
::!]

_output_shapes	
::.^*
(
_output_shapes
::!_

_output_shapes	
::!`

_output_shapes	
::!a

_output_shapes	
::&b"
 
_output_shapes
:
: c

_output_shapes
::d

_output_shapes
: 
É

¯
G__inference_sequential_6_layer_call_and_return_conditional_losses_27200
flatten_input
dense_27194
dense_27196
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
B__inference_flatten_layer_call_and_return_conditional_losses_271642
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_27194dense_27196*
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
@__inference_dense_layer_call_and_return_conditional_losses_271832
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
G__inference_activation_1_layer_call_and_return_conditional_losses_25266

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
Ð/

G__inference_sequential_5_layer_call_and_return_conditional_losses_29406

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
°
c
G__inference_activation_3_layer_call_and_return_conditional_losses_25946

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
Û

P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26671

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
®
«
C__inference_conv2d_1_layer_call_and_return_conditional_losses_25582

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
®
K
/__inference_max_pooling2d_2_layer_call_fn_26928

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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_269222
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
ï
t
H__inference_concatenate_1_layer_call_and_return_conditional_losses_29247
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
Ã
H
,__inference_activation_4_layer_call_fn_30088

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
G__inference_activation_4_layer_call_and_return_conditional_losses_262962
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
Á
å
G__inference_sequential_5_layer_call_and_return_conditional_losses_27139

inputs
conv2d_5_27122
conv2d_5_27124
batch_normalization_5_27128
batch_normalization_5_27130
batch_normalization_5_27132
batch_normalization_5_27134
identity¢-batch_normalization_5/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_27122conv2d_5_27124*
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_269422"
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
G__inference_activation_6_layer_call_and_return_conditional_losses_269762
activation_6/PartitionedCall»
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0batch_normalization_5_27128batch_normalization_5_27130batch_normalization_5_27132batch_normalization_5_27134*
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
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_270212/
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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_269222!
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
%
³
G__inference_sequential_4_layer_call_and_return_conditional_losses_29331

inputs+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource
identity²
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOp¿
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
2
conv2d_4/Conv2D¨
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
conv2d_4/BiasAddm
activation_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_5/mul/x
activation_5/mulMulactivation_5/mul/x:output:0conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/mulo
activation_5/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
activation_5/Sqrt/xm
activation_5/SqrtSqrtactivation_5/Sqrt/x:output:0*
T0*
_output_shapes
: 2
activation_5/Sqrtm
activation_5/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
activation_5/Pow/y
activation_5/PowPowconv2d_4/BiasAdd:output:0activation_5/Pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/Powq
activation_5/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2
activation_5/mul_1/x
activation_5/mul_1Mulactivation_5/mul_1/x:output:0activation_5/Pow:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/mul_1
activation_5/addAddV2conv2d_4/BiasAdd:output:0activation_5/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/add
activation_5/mul_2Mulactivation_5/Sqrt:y:0activation_5/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/mul_2
activation_5/TanhTanhactivation_5/mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/Tanhq
activation_5/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_5/add_1/x¢
activation_5/add_1AddV2activation_5/add_1/x:output:0activation_5/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/add_1
activation_5/mul_3Mulactivation_5/mul:z:0activation_5/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
activation_5/mul_3·
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batch_normalization_4/ReadVariableOp½
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batch_normalization_4/ReadVariableOp_1ê
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpð
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ã
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_5/mul_3:z:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3
IdentityIdentity*batch_normalization_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@:::::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
 
_user_specified_nameinputs


P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29786

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

½
,__inference_sequential_5_layer_call_fn_29479

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
G__inference_sequential_5_layer_call_and_return_conditional_losses_271392
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
¥
c
G__inference_activation_6_layer_call_and_return_conditional_losses_26976

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

f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_26126

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
À
å
G__inference_sequential_1_layer_call_and_return_conditional_losses_25779

inputs
conv2d_1_25762
conv2d_1_25764
batch_normalization_1_25768
batch_normalization_1_25770
batch_normalization_1_25772
batch_normalization_1_25774
identity¢-batch_normalization_1/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_25762conv2d_1_25764*
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_255822"
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
G__inference_activation_2_layer_call_and_return_conditional_losses_256162
activation_2/PartitionedCall¼
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0batch_normalization_1_25768batch_normalization_1_25770batch_normalization_1_25772batch_normalization_1_25774*
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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_256612/
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_255622
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
Ð/

G__inference_sequential_3_layer_call_and_return_conditional_losses_29167

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
Î/

G__inference_sequential_1_layer_call_and_return_conditional_losses_28928

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
Ä
í
G__inference_sequential_2_layer_call_and_return_conditional_losses_26028
conv2d_2_input
conv2d_2_25923
conv2d_2_25925
batch_normalization_2_26018
batch_normalization_2_26020
batch_normalization_2_26022
batch_normalization_2_26024
identity¢-batch_normalization_2/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¦
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_25923conv2d_2_25925*
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_259122"
 conv2d_2/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
G__inference_activation_3_layer_call_and_return_conditional_losses_259462
activation_3/PartitionedCallº
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0batch_normalization_2_26018batch_normalization_2_26020batch_normalization_2_26022batch_normalization_2_26024*
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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_259732/
-batch_normalization_2/StatefulPartitionedCallç
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_2_input
Ú
¨
5__inference_batch_normalization_5_layer_call_fn_30543

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
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_270032
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
¢
Å
,__inference_sequential_4_layer_call_fn_26800
conv2d_4_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¿
StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_267852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
(
_user_specified_nameconv2d_4_input
¿
å
G__inference_sequential_5_layer_call_and_return_conditional_losses_27102

inputs
conv2d_5_27085
conv2d_5_27087
batch_normalization_5_27091
batch_normalization_5_27093
batch_normalization_5_27095
batch_normalization_5_27097
identity¢-batch_normalization_5/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_27085conv2d_5_27087*
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_269422"
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
G__inference_activation_6_layer_call_and_return_conditional_losses_269762
activation_6/PartitionedCall¹
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0batch_normalization_5_27091batch_normalization_5_27093batch_normalization_5_27095batch_normalization_5_27097*
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
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_270032/
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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_269222!
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
¥
c
G__inference_activation_6_layer_call_and_return_conditional_losses_30423

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
´

¨
G__inference_sequential_6_layer_call_and_return_conditional_losses_27242

inputs
dense_27236
dense_27238
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
B__inference_flatten_layer_call_and_return_conditional_losses_271642
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_27236dense_27238*
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
@__inference_dense_layer_call_and_return_conditional_losses_271832
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
¤

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_30466

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
Ú
W
+__inference_concatenate_layer_call_fn_29014
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
F__inference_concatenate_layer_call_and_return_conditional_losses_273552
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

¨
5__inference_batch_normalization_3_layer_call_fn_30139

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
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_261942
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
¥
c
G__inference_activation_5_layer_call_and_return_conditional_losses_26626

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
°
c
G__inference_activation_2_layer_call_and_return_conditional_losses_25616

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
-
ñ
E__inference_sequential_layer_call_and_return_conditional_losses_28815

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
¢
¨
5__inference_batch_normalization_4_layer_call_fn_30309

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
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_265362
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
,__inference_sequential_3_layer_call_fn_29240

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
G__inference_sequential_3_layer_call_and_return_conditional_losses_264592
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
Ú
¦
3__inference_batch_normalization_layer_call_fn_29693

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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_252932
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
È
­
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29768

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
ó
t
H__inference_concatenate_2_layer_call_and_return_conditional_losses_29486
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
À
í
G__inference_sequential_4_layer_call_and_return_conditional_losses_26727
conv2d_4_input
conv2d_4_26711
conv2d_4_26713
batch_normalization_4_26717
batch_normalization_4_26719
batch_normalization_4_26721
batch_normalization_4_26723
identity¢-batch_normalization_4/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¥
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputconv2d_4_26711conv2d_4_26713*
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_265922"
 conv2d_4/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
G__inference_activation_5_layer_call_and_return_conditional_losses_266262
activation_5/PartitionedCall»
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0batch_normalization_4_26717batch_normalization_4_26719batch_normalization_4_26721batch_normalization_4_26723*
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
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_266712/
-batch_normalization_4/StatefulPartitionedCallæ
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
(
_user_specified_nameconv2d_4_input

¨
5__inference_batch_normalization_2_layer_call_fn_30033

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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_258562
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
È
­
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_25856

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

«
N__inference_batch_normalization_layer_call_and_return_conditional_losses_25293

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
þ
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25562

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


,__inference_sequential_6_layer_call_fn_27230
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_272232
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
G__inference_activation_2_layer_call_and_return_conditional_losses_29743

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
 
¨
5__inference_batch_normalization_1_layer_call_fn_29812

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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_255452
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
¬
«
C__inference_conv2d_5_layer_call_and_return_conditional_losses_30396

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
°
c
G__inference_activation_3_layer_call_and_return_conditional_losses_29913

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

}
(__inference_conv2d_5_layer_call_fn_30405

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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_269422
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

­
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29832

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
¨E
ù	
G__inference_functional_1_layer_call_and_return_conditional_losses_27799

inputs
sequential_27709
sequential_27711
sequential_27713
sequential_27715
sequential_27717
sequential_27719
sequential_1_27723
sequential_1_27725
sequential_1_27727
sequential_1_27729
sequential_1_27731
sequential_1_27733
sequential_2_27737
sequential_2_27739
sequential_2_27741
sequential_2_27743
sequential_2_27745
sequential_2_27747
sequential_3_27751
sequential_3_27753
sequential_3_27755
sequential_3_27757
sequential_3_27759
sequential_3_27761
sequential_4_27765
sequential_4_27767
sequential_4_27769
sequential_4_27771
sequential_4_27773
sequential_4_27775
sequential_5_27779
sequential_5_27781
sequential_5_27783
sequential_5_27785
sequential_5_27787
sequential_5_27789
sequential_6_27793
sequential_6_27795
identity¢"sequential/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCall¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall¢$sequential_4/StatefulPartitionedCall¢$sequential_5/StatefulPartitionedCall¢$sequential_6/StatefulPartitionedCallö
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_27709sequential_27711sequential_27713sequential_27715sequential_27717sequential_27719*
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
E__inference_sequential_layer_call_and_return_conditional_losses_253892$
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
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_254462!
max_pooling2d_3/PartitionedCall­
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_27723sequential_1_27725sequential_1_27727sequential_1_27729sequential_1_27731sequential_1_27733*
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_257422&
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
F__inference_concatenate_layer_call_and_return_conditional_losses_273552
concatenate/PartitionedCall¦
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sequential_2_27737sequential_2_27739sequential_2_27741sequential_2_27743sequential_2_27745sequential_2_27747*
Tin
	2*
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
GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_260692&
$sequential_2/StatefulPartitionedCall
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
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_261262!
max_pooling2d_4/PartitionedCall­
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_27751sequential_3_27753sequential_3_27755sequential_3_27757sequential_3_27759sequential_3_27761*
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
G__inference_sequential_3_layer_call_and_return_conditional_losses_264222&
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
H__inference_concatenate_1_layer_call_and_return_conditional_losses_274662
concatenate_1/PartitionedCall§
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0sequential_4_27765sequential_4_27767sequential_4_27769sequential_4_27771sequential_4_27773sequential_4_27775*
Tin
	2*
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
GPU2*0J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_267492&
$sequential_4/StatefulPartitionedCall
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
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_268062!
max_pooling2d_5/PartitionedCall®
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0sequential_5_27779sequential_5_27781sequential_5_27783sequential_5_27785sequential_5_27787sequential_5_27789*
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
G__inference_sequential_5_layer_call_and_return_conditional_losses_271022&
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
H__inference_concatenate_2_layer_call_and_return_conditional_losses_275772
concatenate_2/PartitionedCallÈ
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0sequential_6_27793sequential_6_27795*
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_272232&
$sequential_6/StatefulPartitionedCall
IdentityIdentity-sequential_6/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ê
_input_shapes¸
µ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
¹
,__inference_functional_1_layer_call_fn_28052
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

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity¢StatefulPartitionedCallê
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"#$%&*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_279732
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ê
_input_shapes¸
µ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

¦
3__inference_batch_normalization_layer_call_fn_29629

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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_251762
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
§
C
'__inference_flatten_layer_call_fn_30567

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
B__inference_flatten_layer_call_and_return_conditional_losses_271642
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

½
,__inference_sequential_1_layer_call_fn_29001

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
G__inference_sequential_1_layer_call_and_return_conditional_losses_257792
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
Þ
¨
5__inference_batch_normalization_3_layer_call_fn_30203

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
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_263232
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
Ø
Y
-__inference_concatenate_2_layer_call_fn_29492
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
H__inference_concatenate_2_layer_call_and_return_conditional_losses_275772
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

¦
3__inference_batch_normalization_layer_call_fn_29642

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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_252072
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
Â
»
G__inference_sequential_6_layer_call_and_return_conditional_losses_29505

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


N__inference_batch_normalization_layer_call_and_return_conditional_losses_29616

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
¿
H
,__inference_activation_6_layer_call_fn_30428

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
G__inference_activation_6_layer_call_and_return_conditional_losses_269762
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
.

G__inference_sequential_2_layer_call_and_return_conditional_losses_29054

inputs+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource
identity¢$batch_normalization_2/AssignNewValue¢&batch_normalization_2/AssignNewValue_1°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpÀ
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp®
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_2/BiasAddm
activation_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
activation_3/mul/x
activation_3/mulMulactivation_3/mul/x:output:0conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/mulo
activation_3/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2
activation_3/Sqrt/xm
activation_3/SqrtSqrtactivation_3/Sqrt/x:output:0*
T0*
_output_shapes
: 2
activation_3/Sqrtm
activation_3/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2
activation_3/Pow/y
activation_3/PowPowconv2d_2/BiasAdd:output:0activation_3/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/Powq
activation_3/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2
activation_3/mul_1/x 
activation_3/mul_1Mulactivation_3/mul_1/x:output:0activation_3/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/mul_1
activation_3/addAddV2conv2d_2/BiasAdd:output:0activation_3/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/add
activation_3/mul_2Mulactivation_3/Sqrt:y:0activation_3/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/mul_2
activation_3/TanhTanhactivation_3/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/Tanhq
activation_3/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
activation_3/add_1/x£
activation_3/add_1AddV2activation_3/add_1/x:output:0activation_3/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/add_1
activation_3/mul_3Mulactivation_3/mul:z:0activation_3/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
activation_3/mul_3¶
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp¼
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1é
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpï
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1î
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_3/mul_3:z:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2(
&batch_normalization_2/FusedBatchNormV3
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*Q
_classG
ECloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValue
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*S
_classI
GEloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1Ø
IdentityIdentity*batch_normalization_2/FusedBatchNormV3:y:0%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_1:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
´

¨
G__inference_sequential_6_layer_call_and_return_conditional_losses_27223

inputs
dense_27217
dense_27219
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
B__inference_flatten_layer_call_and_return_conditional_losses_271642
flatten/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_27217dense_27219*
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
@__inference_dense_layer_call_and_return_conditional_losses_271832
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

f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_25446

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
ª
¹
,__inference_functional_1_layer_call_fn_27878
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

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36
identity¢StatefulPartitionedCallÞ
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
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36*2
Tin+
)2'*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
 !"%&*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_277992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ê
_input_shapes¸
µ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

}
(__inference_conv2d_4_layer_call_fn_30235

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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_265922
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

½
,__inference_sequential_2_layer_call_fn_29126

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
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_261052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ú
¨
5__inference_batch_normalization_4_layer_call_fn_30373

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
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_266532
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
Ã
H
,__inference_activation_3_layer_call_fn_29918

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
G__inference_activation_3_layer_call_and_return_conditional_losses_259462
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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_26922

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
¬
å
G__inference_sequential_2_layer_call_and_return_conditional_losses_26069

inputs
conv2d_2_26053
conv2d_2_26055
batch_normalization_2_26059
batch_normalization_2_26061
batch_normalization_2_26063
batch_normalization_2_26065
identity¢-batch_normalization_2/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_26053conv2d_2_26055*
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_259122"
 conv2d_2/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
G__inference_activation_3_layer_call_and_return_conditional_losses_259462
activation_3/PartitionedCallº
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0batch_normalization_2_26059batch_normalization_2_26061batch_normalization_2_26063batch_normalization_2_26065*
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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_259732/
-batch_normalization_2/StatefulPartitionedCallç
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

­
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_30172

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
Ô
­
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_30278

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


P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_30126

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
×
í
G__inference_sequential_5_layer_call_and_return_conditional_losses_27059
conv2d_5_input
conv2d_5_26953
conv2d_5_26955
batch_normalization_5_27048
batch_normalization_5_27050
batch_normalization_5_27052
batch_normalization_5_27054
identity¢-batch_normalization_5/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall¥
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_26953conv2d_5_26955*
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_269422"
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
G__inference_activation_6_layer_call_and_return_conditional_losses_269762
activation_6/PartitionedCall¹
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0batch_normalization_5_27048batch_normalization_5_27050batch_normalization_5_27052batch_normalization_5_27054*
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
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_270032/
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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_269222!
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
®
«
C__inference_conv2d_2_layer_call_and_return_conditional_losses_25912

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
Ú
í
G__inference_sequential_3_layer_call_and_return_conditional_losses_26379
conv2d_3_input
conv2d_3_26273
conv2d_3_26275
batch_normalization_3_26368
batch_normalization_3_26370
batch_normalization_3_26372
batch_normalization_3_26374
identity¢-batch_normalization_3/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¦
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallconv2d_3_inputconv2d_3_26273conv2d_3_26275*
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
C__inference_conv2d_3_layer_call_and_return_conditional_losses_262622"
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
G__inference_activation_4_layer_call_and_return_conditional_losses_262962
activation_4/PartitionedCallº
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0batch_normalization_3_26368batch_normalization_3_26370batch_normalization_3_26372batch_normalization_3_26374*
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
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_263232/
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_262422!
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
¾
^
B__inference_flatten_layer_call_and_return_conditional_losses_30562

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
¬
©
A__inference_conv2d_layer_call_and_return_conditional_losses_29546

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
 
¨
5__inference_batch_normalization_3_layer_call_fn_30152

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
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_262252
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
§
ß
G__inference_functional_1_layer_call_and_return_conditional_losses_28613

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
4sequential_2_conv2d_2_conv2d_readvariableop_resource9
5sequential_2_conv2d_2_biasadd_readvariableop_resource>
:sequential_2_batch_normalization_2_readvariableop_resource@
<sequential_2_batch_normalization_2_readvariableop_1_resourceO
Ksequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceQ
Msequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource8
4sequential_3_conv2d_3_conv2d_readvariableop_resource9
5sequential_3_conv2d_3_biasadd_readvariableop_resource>
:sequential_3_batch_normalization_3_readvariableop_resource@
<sequential_3_batch_normalization_3_readvariableop_1_resourceO
Ksequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceQ
Msequential_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource8
4sequential_4_conv2d_4_conv2d_readvariableop_resource9
5sequential_4_conv2d_4_biasadd_readvariableop_resource>
:sequential_4_batch_normalization_4_readvariableop_resource@
<sequential_4_batch_normalization_4_readvariableop_1_resourceO
Ksequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceQ
Msequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource8
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
concatenate/concat×
+sequential_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4sequential_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+sequential_2/conv2d_2/Conv2D/ReadVariableOpü
sequential_2/conv2d_2/Conv2DConv2Dconcatenate/concat:output:03sequential_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
sequential_2/conv2d_2/Conv2DÎ
,sequential_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5sequential_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,sequential_2/conv2d_2/BiasAdd/ReadVariableOpâ
sequential_2/conv2d_2/BiasAddBiasAdd%sequential_2/conv2d_2/Conv2D:output:04sequential_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_2/conv2d_2/BiasAdd
sequential_2/activation_3/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential_2/activation_3/mul/xÓ
sequential_2/activation_3/mulMul(sequential_2/activation_3/mul/x:output:0&sequential_2/conv2d_2/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_2/activation_3/mul
 sequential_2/activation_3/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2"
 sequential_2/activation_3/Sqrt/x
sequential_2/activation_3/SqrtSqrt)sequential_2/activation_3/Sqrt/x:output:0*
T0*
_output_shapes
: 2 
sequential_2/activation_3/Sqrt
sequential_2/activation_3/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2!
sequential_2/activation_3/Pow/yÓ
sequential_2/activation_3/PowPow&sequential_2/conv2d_2/BiasAdd:output:0(sequential_2/activation_3/Pow/y:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_2/activation_3/Pow
!sequential_2/activation_3/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2#
!sequential_2/activation_3/mul_1/xÔ
sequential_2/activation_3/mul_1Mul*sequential_2/activation_3/mul_1/x:output:0!sequential_2/activation_3/Pow:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_2/activation_3/mul_1Ð
sequential_2/activation_3/addAddV2&sequential_2/conv2d_2/BiasAdd:output:0#sequential_2/activation_3/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_2/activation_3/addÌ
sequential_2/activation_3/mul_2Mul"sequential_2/activation_3/Sqrt:y:0!sequential_2/activation_3/add:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_2/activation_3/mul_2©
sequential_2/activation_3/TanhTanh#sequential_2/activation_3/mul_2:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
sequential_2/activation_3/Tanh
!sequential_2/activation_3/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!sequential_2/activation_3/add_1/x×
sequential_2/activation_3/add_1AddV2*sequential_2/activation_3/add_1/x:output:0"sequential_2/activation_3/Tanh:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_2/activation_3/add_1Í
sequential_2/activation_3/mul_3Mul!sequential_2/activation_3/mul:z:0#sequential_2/activation_3/add_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_2/activation_3/mul_3Ý
1sequential_2/batch_normalization_2/ReadVariableOpReadVariableOp:sequential_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype023
1sequential_2/batch_normalization_2/ReadVariableOpã
3sequential_2/batch_normalization_2/ReadVariableOp_1ReadVariableOp<sequential_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype025
3sequential_2/batch_normalization_2/ReadVariableOp_1
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02D
Bsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02F
Dsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1»
3sequential_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3#sequential_2/activation_3/mul_3:z:09sequential_2/batch_normalization_2/ReadVariableOp:value:0;sequential_2/batch_normalization_2/ReadVariableOp_1:value:0Jsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 25
3sequential_2/batch_normalization_2/FusedBatchNormV3Ç
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
+sequential_3/conv2d_3/Conv2D/ReadVariableOp
sequential_3/conv2d_3/Conv2DConv2D7sequential_2/batch_normalization_2/FusedBatchNormV3:y:03sequential_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
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
concatenate_1/concatÙ
+sequential_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4sequential_4_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+sequential_4/conv2d_4/Conv2D/ReadVariableOpý
sequential_4/conv2d_4/Conv2DConv2Dconcatenate_1/concat:output:03sequential_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*
paddingSAME*
strides
2
sequential_4/conv2d_4/Conv2DÏ
,sequential_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,sequential_4/conv2d_4/BiasAdd/ReadVariableOpá
sequential_4/conv2d_4/BiasAddBiasAdd%sequential_4/conv2d_4/Conv2D:output:04sequential_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_4/conv2d_4/BiasAdd
sequential_4/activation_5/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
sequential_4/activation_5/mul/xÒ
sequential_4/activation_5/mulMul(sequential_4/activation_5/mul/x:output:0&sequential_4/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_4/activation_5/mul
 sequential_4/activation_5/Sqrt/xConst*
_output_shapes
: *
dtype0*
valueB
 *ù"?2"
 sequential_4/activation_5/Sqrt/x
sequential_4/activation_5/SqrtSqrt)sequential_4/activation_5/Sqrt/x:output:0*
T0*
_output_shapes
: 2 
sequential_4/activation_5/Sqrt
sequential_4/activation_5/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@2!
sequential_4/activation_5/Pow/yÒ
sequential_4/activation_5/PowPow&sequential_4/conv2d_4/BiasAdd:output:0(sequential_4/activation_5/Pow/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_4/activation_5/Pow
!sequential_4/activation_5/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=2#
!sequential_4/activation_5/mul_1/xÓ
sequential_4/activation_5/mul_1Mul*sequential_4/activation_5/mul_1/x:output:0!sequential_4/activation_5/Pow:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_4/activation_5/mul_1Ï
sequential_4/activation_5/addAddV2&sequential_4/conv2d_4/BiasAdd:output:0#sequential_4/activation_5/mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2
sequential_4/activation_5/addË
sequential_4/activation_5/mul_2Mul"sequential_4/activation_5/Sqrt:y:0!sequential_4/activation_5/add:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_4/activation_5/mul_2¨
sequential_4/activation_5/TanhTanh#sequential_4/activation_5/mul_2:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2 
sequential_4/activation_5/Tanh
!sequential_4/activation_5/add_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!sequential_4/activation_5/add_1/xÖ
sequential_4/activation_5/add_1AddV2*sequential_4/activation_5/add_1/x:output:0"sequential_4/activation_5/Tanh:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_4/activation_5/add_1Ì
sequential_4/activation_5/mul_3Mul!sequential_4/activation_5/mul:z:0#sequential_4/activation_5/add_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2!
sequential_4/activation_5/mul_3Þ
1sequential_4/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_4_batch_normalization_4_readvariableop_resource*
_output_shapes	
:*
dtype023
1sequential_4/batch_normalization_4/ReadVariableOpä
3sequential_4/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_4_batch_normalization_4_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3sequential_4/batch_normalization_4/ReadVariableOp_1
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02D
Bsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_4_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02F
Dsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1¾
3sequential_4/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3#sequential_4/activation_5/mul_3:z:09sequential_4/batch_normalization_4/ReadVariableOp:value:0;sequential_4/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_4/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ@@:::::*
epsilon%o:*
is_training( 25
3sequential_4/batch_normalization_4/FusedBatchNormV3Ê
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
+sequential_5/conv2d_5/Conv2D/ReadVariableOp
sequential_5/conv2d_5/Conv2DConv2D7sequential_4/batch_normalization_4/FusedBatchNormV3:y:03sequential_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
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
identityIdentity:output:0*Ê
_input_shapes¸
µ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::::::::::::::::::::::Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾
^
B__inference_flatten_layer_call_and_return_conditional_losses_27164

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
Ô
Y
-__inference_concatenate_1_layer_call_fn_29253
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
H__inference_concatenate_1_layer_call_and_return_conditional_losses_274662
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
³
¨
@__inference_dense_layer_call_and_return_conditional_losses_27183

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
ç
r
H__inference_concatenate_1_layer_call_and_return_conditional_losses_27466

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
Õ

N__inference_batch_normalization_layer_call_and_return_conditional_losses_25311

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
ý«
ò@
!__inference__traced_restore_31214
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate/
+assignvariableop_5_sequential_conv2d_kernel-
)assignvariableop_6_sequential_conv2d_bias;
7assignvariableop_7_sequential_batch_normalization_gamma:
6assignvariableop_8_sequential_batch_normalization_betaA
=assignvariableop_9_sequential_batch_normalization_moving_meanF
Bassignvariableop_10_sequential_batch_normalization_moving_variance4
0assignvariableop_11_sequential_1_conv2d_1_kernel2
.assignvariableop_12_sequential_1_conv2d_1_bias@
<assignvariableop_13_sequential_1_batch_normalization_1_gamma?
;assignvariableop_14_sequential_1_batch_normalization_1_betaF
Bassignvariableop_15_sequential_1_batch_normalization_1_moving_meanJ
Fassignvariableop_16_sequential_1_batch_normalization_1_moving_variance4
0assignvariableop_17_sequential_2_conv2d_2_kernel2
.assignvariableop_18_sequential_2_conv2d_2_bias@
<assignvariableop_19_sequential_2_batch_normalization_2_gamma?
;assignvariableop_20_sequential_2_batch_normalization_2_betaF
Bassignvariableop_21_sequential_2_batch_normalization_2_moving_meanJ
Fassignvariableop_22_sequential_2_batch_normalization_2_moving_variance4
0assignvariableop_23_sequential_3_conv2d_3_kernel2
.assignvariableop_24_sequential_3_conv2d_3_bias@
<assignvariableop_25_sequential_3_batch_normalization_3_gamma?
;assignvariableop_26_sequential_3_batch_normalization_3_betaF
Bassignvariableop_27_sequential_3_batch_normalization_3_moving_meanJ
Fassignvariableop_28_sequential_3_batch_normalization_3_moving_variance4
0assignvariableop_29_sequential_4_conv2d_4_kernel2
.assignvariableop_30_sequential_4_conv2d_4_bias@
<assignvariableop_31_sequential_4_batch_normalization_4_gamma?
;assignvariableop_32_sequential_4_batch_normalization_4_betaF
Bassignvariableop_33_sequential_4_batch_normalization_4_moving_meanJ
Fassignvariableop_34_sequential_4_batch_normalization_4_moving_variance4
0assignvariableop_35_sequential_5_conv2d_5_kernel2
.assignvariableop_36_sequential_5_conv2d_5_bias@
<assignvariableop_37_sequential_5_batch_normalization_5_gamma?
;assignvariableop_38_sequential_5_batch_normalization_5_betaF
Bassignvariableop_39_sequential_5_batch_normalization_5_moving_meanJ
Fassignvariableop_40_sequential_5_batch_normalization_5_moving_variance1
-assignvariableop_41_sequential_6_dense_kernel/
+assignvariableop_42_sequential_6_dense_bias
assignvariableop_43_total
assignvariableop_44_count
assignvariableop_45_total_1
assignvariableop_46_count_17
3assignvariableop_47_adam_sequential_conv2d_kernel_m5
1assignvariableop_48_adam_sequential_conv2d_bias_mC
?assignvariableop_49_adam_sequential_batch_normalization_gamma_mB
>assignvariableop_50_adam_sequential_batch_normalization_beta_m;
7assignvariableop_51_adam_sequential_1_conv2d_1_kernel_m9
5assignvariableop_52_adam_sequential_1_conv2d_1_bias_mG
Cassignvariableop_53_adam_sequential_1_batch_normalization_1_gamma_mF
Bassignvariableop_54_adam_sequential_1_batch_normalization_1_beta_m;
7assignvariableop_55_adam_sequential_2_conv2d_2_kernel_m9
5assignvariableop_56_adam_sequential_2_conv2d_2_bias_mG
Cassignvariableop_57_adam_sequential_2_batch_normalization_2_gamma_mF
Bassignvariableop_58_adam_sequential_2_batch_normalization_2_beta_m;
7assignvariableop_59_adam_sequential_3_conv2d_3_kernel_m9
5assignvariableop_60_adam_sequential_3_conv2d_3_bias_mG
Cassignvariableop_61_adam_sequential_3_batch_normalization_3_gamma_mF
Bassignvariableop_62_adam_sequential_3_batch_normalization_3_beta_m;
7assignvariableop_63_adam_sequential_4_conv2d_4_kernel_m9
5assignvariableop_64_adam_sequential_4_conv2d_4_bias_mG
Cassignvariableop_65_adam_sequential_4_batch_normalization_4_gamma_mF
Bassignvariableop_66_adam_sequential_4_batch_normalization_4_beta_m;
7assignvariableop_67_adam_sequential_5_conv2d_5_kernel_m9
5assignvariableop_68_adam_sequential_5_conv2d_5_bias_mG
Cassignvariableop_69_adam_sequential_5_batch_normalization_5_gamma_mF
Bassignvariableop_70_adam_sequential_5_batch_normalization_5_beta_m8
4assignvariableop_71_adam_sequential_6_dense_kernel_m6
2assignvariableop_72_adam_sequential_6_dense_bias_m7
3assignvariableop_73_adam_sequential_conv2d_kernel_v5
1assignvariableop_74_adam_sequential_conv2d_bias_vC
?assignvariableop_75_adam_sequential_batch_normalization_gamma_vB
>assignvariableop_76_adam_sequential_batch_normalization_beta_v;
7assignvariableop_77_adam_sequential_1_conv2d_1_kernel_v9
5assignvariableop_78_adam_sequential_1_conv2d_1_bias_vG
Cassignvariableop_79_adam_sequential_1_batch_normalization_1_gamma_vF
Bassignvariableop_80_adam_sequential_1_batch_normalization_1_beta_v;
7assignvariableop_81_adam_sequential_2_conv2d_2_kernel_v9
5assignvariableop_82_adam_sequential_2_conv2d_2_bias_vG
Cassignvariableop_83_adam_sequential_2_batch_normalization_2_gamma_vF
Bassignvariableop_84_adam_sequential_2_batch_normalization_2_beta_v;
7assignvariableop_85_adam_sequential_3_conv2d_3_kernel_v9
5assignvariableop_86_adam_sequential_3_conv2d_3_bias_vG
Cassignvariableop_87_adam_sequential_3_batch_normalization_3_gamma_vF
Bassignvariableop_88_adam_sequential_3_batch_normalization_3_beta_v;
7assignvariableop_89_adam_sequential_4_conv2d_4_kernel_v9
5assignvariableop_90_adam_sequential_4_conv2d_4_bias_vG
Cassignvariableop_91_adam_sequential_4_batch_normalization_4_gamma_vF
Bassignvariableop_92_adam_sequential_4_batch_normalization_4_beta_v;
7assignvariableop_93_adam_sequential_5_conv2d_5_kernel_v9
5assignvariableop_94_adam_sequential_5_conv2d_5_bias_vG
Cassignvariableop_95_adam_sequential_5_batch_normalization_5_gamma_vF
Bassignvariableop_96_adam_sequential_5_batch_normalization_5_beta_v8
4assignvariableop_97_adam_sequential_6_dense_kernel_v6
2assignvariableop_98_adam_sequential_6_dense_bias_v
identity_100¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98ä,
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*ð+
valueæ+Bã+dB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB'variables/34/.ATTRIBUTES/VARIABLE_VALUEB'variables/35/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/30/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/31/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/32/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/33/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/36/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/37/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÙ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:d*
dtype0*Ý
valueÓBÐdB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices¢
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¦
_output_shapes
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*r
dtypesh
f2d	2
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

Identity_9Â
AssignVariableOp_9AssignVariableOp=assignvariableop_9_sequential_batch_normalization_moving_meanIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ê
AssignVariableOp_10AssignVariableOpBassignvariableop_10_sequential_batch_normalization_moving_varianceIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¸
AssignVariableOp_11AssignVariableOp0assignvariableop_11_sequential_1_conv2d_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¶
AssignVariableOp_12AssignVariableOp.assignvariableop_12_sequential_1_conv2d_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Ä
AssignVariableOp_13AssignVariableOp<assignvariableop_13_sequential_1_batch_normalization_1_gammaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ã
AssignVariableOp_14AssignVariableOp;assignvariableop_14_sequential_1_batch_normalization_1_betaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ê
AssignVariableOp_15AssignVariableOpBassignvariableop_15_sequential_1_batch_normalization_1_moving_meanIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Î
AssignVariableOp_16AssignVariableOpFassignvariableop_16_sequential_1_batch_normalization_1_moving_varianceIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¸
AssignVariableOp_17AssignVariableOp0assignvariableop_17_sequential_2_conv2d_2_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¶
AssignVariableOp_18AssignVariableOp.assignvariableop_18_sequential_2_conv2d_2_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Ä
AssignVariableOp_19AssignVariableOp<assignvariableop_19_sequential_2_batch_normalization_2_gammaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ã
AssignVariableOp_20AssignVariableOp;assignvariableop_20_sequential_2_batch_normalization_2_betaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21Ê
AssignVariableOp_21AssignVariableOpBassignvariableop_21_sequential_2_batch_normalization_2_moving_meanIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Î
AssignVariableOp_22AssignVariableOpFassignvariableop_22_sequential_2_batch_normalization_2_moving_varianceIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¸
AssignVariableOp_23AssignVariableOp0assignvariableop_23_sequential_3_conv2d_3_kernelIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¶
AssignVariableOp_24AssignVariableOp.assignvariableop_24_sequential_3_conv2d_3_biasIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ä
AssignVariableOp_25AssignVariableOp<assignvariableop_25_sequential_3_batch_normalization_3_gammaIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Ã
AssignVariableOp_26AssignVariableOp;assignvariableop_26_sequential_3_batch_normalization_3_betaIdentity_26:output:0"/device:CPU:0*
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
Identity_29¸
AssignVariableOp_29AssignVariableOp0assignvariableop_29_sequential_4_conv2d_4_kernelIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¶
AssignVariableOp_30AssignVariableOp.assignvariableop_30_sequential_4_conv2d_4_biasIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Ä
AssignVariableOp_31AssignVariableOp<assignvariableop_31_sequential_4_batch_normalization_4_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ã
AssignVariableOp_32AssignVariableOp;assignvariableop_32_sequential_4_batch_normalization_4_betaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33Ê
AssignVariableOp_33AssignVariableOpBassignvariableop_33_sequential_4_batch_normalization_4_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Î
AssignVariableOp_34AssignVariableOpFassignvariableop_34_sequential_4_batch_normalization_4_moving_varianceIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35¸
AssignVariableOp_35AssignVariableOp0assignvariableop_35_sequential_5_conv2d_5_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¶
AssignVariableOp_36AssignVariableOp.assignvariableop_36_sequential_5_conv2d_5_biasIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ä
AssignVariableOp_37AssignVariableOp<assignvariableop_37_sequential_5_batch_normalization_5_gammaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38Ã
AssignVariableOp_38AssignVariableOp;assignvariableop_38_sequential_5_batch_normalization_5_betaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Ê
AssignVariableOp_39AssignVariableOpBassignvariableop_39_sequential_5_batch_normalization_5_moving_meanIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40Î
AssignVariableOp_40AssignVariableOpFassignvariableop_40_sequential_5_batch_normalization_5_moving_varianceIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41µ
AssignVariableOp_41AssignVariableOp-assignvariableop_41_sequential_6_dense_kernelIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42³
AssignVariableOp_42AssignVariableOp+assignvariableop_42_sequential_6_dense_biasIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43¡
AssignVariableOp_43AssignVariableOpassignvariableop_43_totalIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¡
AssignVariableOp_44AssignVariableOpassignvariableop_44_countIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45£
AssignVariableOp_45AssignVariableOpassignvariableop_45_total_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46£
AssignVariableOp_46AssignVariableOpassignvariableop_46_count_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47»
AssignVariableOp_47AssignVariableOp3assignvariableop_47_adam_sequential_conv2d_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48¹
AssignVariableOp_48AssignVariableOp1assignvariableop_48_adam_sequential_conv2d_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49Ç
AssignVariableOp_49AssignVariableOp?assignvariableop_49_adam_sequential_batch_normalization_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50Æ
AssignVariableOp_50AssignVariableOp>assignvariableop_50_adam_sequential_batch_normalization_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¿
AssignVariableOp_51AssignVariableOp7assignvariableop_51_adam_sequential_1_conv2d_1_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52½
AssignVariableOp_52AssignVariableOp5assignvariableop_52_adam_sequential_1_conv2d_1_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53Ë
AssignVariableOp_53AssignVariableOpCassignvariableop_53_adam_sequential_1_batch_normalization_1_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54Ê
AssignVariableOp_54AssignVariableOpBassignvariableop_54_adam_sequential_1_batch_normalization_1_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¿
AssignVariableOp_55AssignVariableOp7assignvariableop_55_adam_sequential_2_conv2d_2_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56½
AssignVariableOp_56AssignVariableOp5assignvariableop_56_adam_sequential_2_conv2d_2_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57Ë
AssignVariableOp_57AssignVariableOpCassignvariableop_57_adam_sequential_2_batch_normalization_2_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58Ê
AssignVariableOp_58AssignVariableOpBassignvariableop_58_adam_sequential_2_batch_normalization_2_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59¿
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_sequential_3_conv2d_3_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60½
AssignVariableOp_60AssignVariableOp5assignvariableop_60_adam_sequential_3_conv2d_3_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61Ë
AssignVariableOp_61AssignVariableOpCassignvariableop_61_adam_sequential_3_batch_normalization_3_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62Ê
AssignVariableOp_62AssignVariableOpBassignvariableop_62_adam_sequential_3_batch_normalization_3_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63¿
AssignVariableOp_63AssignVariableOp7assignvariableop_63_adam_sequential_4_conv2d_4_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64½
AssignVariableOp_64AssignVariableOp5assignvariableop_64_adam_sequential_4_conv2d_4_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65Ë
AssignVariableOp_65AssignVariableOpCassignvariableop_65_adam_sequential_4_batch_normalization_4_gamma_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66Ê
AssignVariableOp_66AssignVariableOpBassignvariableop_66_adam_sequential_4_batch_normalization_4_beta_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67¿
AssignVariableOp_67AssignVariableOp7assignvariableop_67_adam_sequential_5_conv2d_5_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68½
AssignVariableOp_68AssignVariableOp5assignvariableop_68_adam_sequential_5_conv2d_5_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69Ë
AssignVariableOp_69AssignVariableOpCassignvariableop_69_adam_sequential_5_batch_normalization_5_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70Ê
AssignVariableOp_70AssignVariableOpBassignvariableop_70_adam_sequential_5_batch_normalization_5_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71¼
AssignVariableOp_71AssignVariableOp4assignvariableop_71_adam_sequential_6_dense_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72º
AssignVariableOp_72AssignVariableOp2assignvariableop_72_adam_sequential_6_dense_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73»
AssignVariableOp_73AssignVariableOp3assignvariableop_73_adam_sequential_conv2d_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74¹
AssignVariableOp_74AssignVariableOp1assignvariableop_74_adam_sequential_conv2d_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75Ç
AssignVariableOp_75AssignVariableOp?assignvariableop_75_adam_sequential_batch_normalization_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76Æ
AssignVariableOp_76AssignVariableOp>assignvariableop_76_adam_sequential_batch_normalization_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77¿
AssignVariableOp_77AssignVariableOp7assignvariableop_77_adam_sequential_1_conv2d_1_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78½
AssignVariableOp_78AssignVariableOp5assignvariableop_78_adam_sequential_1_conv2d_1_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79Ë
AssignVariableOp_79AssignVariableOpCassignvariableop_79_adam_sequential_1_batch_normalization_1_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80Ê
AssignVariableOp_80AssignVariableOpBassignvariableop_80_adam_sequential_1_batch_normalization_1_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81¿
AssignVariableOp_81AssignVariableOp7assignvariableop_81_adam_sequential_2_conv2d_2_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82½
AssignVariableOp_82AssignVariableOp5assignvariableop_82_adam_sequential_2_conv2d_2_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83Ë
AssignVariableOp_83AssignVariableOpCassignvariableop_83_adam_sequential_2_batch_normalization_2_gamma_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84Ê
AssignVariableOp_84AssignVariableOpBassignvariableop_84_adam_sequential_2_batch_normalization_2_beta_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85¿
AssignVariableOp_85AssignVariableOp7assignvariableop_85_adam_sequential_3_conv2d_3_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86½
AssignVariableOp_86AssignVariableOp5assignvariableop_86_adam_sequential_3_conv2d_3_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87Ë
AssignVariableOp_87AssignVariableOpCassignvariableop_87_adam_sequential_3_batch_normalization_3_gamma_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88Ê
AssignVariableOp_88AssignVariableOpBassignvariableop_88_adam_sequential_3_batch_normalization_3_beta_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89¿
AssignVariableOp_89AssignVariableOp7assignvariableop_89_adam_sequential_4_conv2d_4_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90½
AssignVariableOp_90AssignVariableOp5assignvariableop_90_adam_sequential_4_conv2d_4_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91Ë
AssignVariableOp_91AssignVariableOpCassignvariableop_91_adam_sequential_4_batch_normalization_4_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92Ê
AssignVariableOp_92AssignVariableOpBassignvariableop_92_adam_sequential_4_batch_normalization_4_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93¿
AssignVariableOp_93AssignVariableOp7assignvariableop_93_adam_sequential_5_conv2d_5_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94½
AssignVariableOp_94AssignVariableOp5assignvariableop_94_adam_sequential_5_conv2d_5_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95Ë
AssignVariableOp_95AssignVariableOpCassignvariableop_95_adam_sequential_5_batch_normalization_5_gamma_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96Ê
AssignVariableOp_96AssignVariableOpBassignvariableop_96_adam_sequential_5_batch_normalization_5_beta_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97¼
AssignVariableOp_97AssignVariableOp4assignvariableop_97_adam_sequential_6_dense_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98º
AssignVariableOp_98AssignVariableOp2assignvariableop_98_adam_sequential_6_dense_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_989
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpà
Identity_99Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_99Õ
Identity_100IdentityIdentity_99:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98*
T0*
_output_shapes
: 2
Identity_100"%
identity_100Identity_100:output:0*£
_input_shapes
: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_98:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
®
K
/__inference_max_pooling2d_5_layer_call_fn_26812

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
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_268062
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
×

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26341

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
Â
å
G__inference_sequential_3_layer_call_and_return_conditional_losses_26422

inputs
conv2d_3_26405
conv2d_3_26407
batch_normalization_3_26411
batch_normalization_3_26413
batch_normalization_3_26415
batch_normalization_3_26417
identity¢-batch_normalization_3/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_26405conv2d_3_26407*
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
C__inference_conv2d_3_layer_call_and_return_conditional_losses_262622"
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
G__inference_activation_4_layer_call_and_return_conditional_losses_262962
activation_4/PartitionedCallº
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0batch_normalization_3_26411batch_normalization_3_26413batch_normalization_3_26415batch_normalization_3_26417*
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
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_263232/
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_262422!
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
°
c
G__inference_activation_4_layer_call_and_return_conditional_losses_30083

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
Ü
¦
3__inference_batch_normalization_layer_call_fn_29706

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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_253112
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
 
Å
,__inference_sequential_3_layer_call_fn_26437
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
G__inference_sequential_3_layer_call_and_return_conditional_losses_264222
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
¤
Å
,__inference_sequential_1_layer_call_fn_25757
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_257422
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
×

P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_30190

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

½
,__inference_sequential_2_layer_call_fn_29109

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
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_260692
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

«
N__inference_batch_normalization_layer_call_and_return_conditional_losses_29662

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

Ù
E__inference_sequential_layer_call_and_return_conditional_losses_25348
conv2d_input
conv2d_25243
conv2d_25245
batch_normalization_25338
batch_normalization_25340
batch_normalization_25342
batch_normalization_25344
identity¢+batch_normalization/StatefulPartitionedCall¢conv2d/StatefulPartitionedCall
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_25243conv2d_25245*
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
A__inference_conv2d_layer_call_and_return_conditional_losses_252322 
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
G__inference_activation_1_layer_call_and_return_conditional_losses_252662
activation_1/PartitionedCall¬
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:0batch_normalization_25338batch_normalization_25340batch_normalization_25342batch_normalization_25344*
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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_252932-
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
¢
Å
,__inference_sequential_3_layer_call_fn_26474
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
G__inference_sequential_3_layer_call_and_return_conditional_losses_264592
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
È
­
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_30002

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
Ü
¨
5__inference_batch_normalization_4_layer_call_fn_30386

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
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_266712
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
 
Å
,__inference_sequential_5_layer_call_fn_27117
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
G__inference_sequential_5_layer_call_and_return_conditional_losses_271022
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
¢
¨
5__inference_batch_normalization_5_layer_call_fn_30479

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
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_268742
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
«E
ú	
G__inference_functional_1_layer_call_and_return_conditional_losses_27610
input_1
sequential_27287
sequential_27289
sequential_27291
sequential_27293
sequential_27295
sequential_27297
sequential_1_27335
sequential_1_27337
sequential_1_27339
sequential_1_27341
sequential_1_27343
sequential_1_27345
sequential_2_27398
sequential_2_27400
sequential_2_27402
sequential_2_27404
sequential_2_27406
sequential_2_27408
sequential_3_27446
sequential_3_27448
sequential_3_27450
sequential_3_27452
sequential_3_27454
sequential_3_27456
sequential_4_27509
sequential_4_27511
sequential_4_27513
sequential_4_27515
sequential_4_27517
sequential_4_27519
sequential_5_27557
sequential_5_27559
sequential_5_27561
sequential_5_27563
sequential_5_27565
sequential_5_27567
sequential_6_27604
sequential_6_27606
identity¢"sequential/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCall¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall¢$sequential_4/StatefulPartitionedCall¢$sequential_5/StatefulPartitionedCall¢$sequential_6/StatefulPartitionedCall÷
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_27287sequential_27289sequential_27291sequential_27293sequential_27295sequential_27297*
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
E__inference_sequential_layer_call_and_return_conditional_losses_253892$
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
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_254462!
max_pooling2d_3/PartitionedCall­
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_27335sequential_1_27337sequential_1_27339sequential_1_27341sequential_1_27343sequential_1_27345*
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_257422&
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
F__inference_concatenate_layer_call_and_return_conditional_losses_273552
concatenate/PartitionedCall¦
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sequential_2_27398sequential_2_27400sequential_2_27402sequential_2_27404sequential_2_27406sequential_2_27408*
Tin
	2*
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
GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_260692&
$sequential_2/StatefulPartitionedCall
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
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_261262!
max_pooling2d_4/PartitionedCall­
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_27446sequential_3_27448sequential_3_27450sequential_3_27452sequential_3_27454sequential_3_27456*
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
G__inference_sequential_3_layer_call_and_return_conditional_losses_264222&
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
H__inference_concatenate_1_layer_call_and_return_conditional_losses_274662
concatenate_1/PartitionedCall§
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0sequential_4_27509sequential_4_27511sequential_4_27513sequential_4_27515sequential_4_27517sequential_4_27519*
Tin
	2*
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
GPU2*0J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_267492&
$sequential_4/StatefulPartitionedCall
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
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_268062!
max_pooling2d_5/PartitionedCall®
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0sequential_5_27557sequential_5_27559sequential_5_27561sequential_5_27563sequential_5_27565sequential_5_27567*
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
G__inference_sequential_5_layer_call_and_return_conditional_losses_271022&
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
H__inference_concatenate_2_layer_call_and_return_conditional_losses_275772
concatenate_2/PartitionedCallÈ
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0sequential_6_27604sequential_6_27606*
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_272232&
$sequential_6/StatefulPartitionedCall
IdentityIdentity-sequential_6/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ê
_input_shapes¸
µ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
¬
«
C__inference_conv2d_4_layer_call_and_return_conditional_losses_26592

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
¤
¨
5__inference_batch_normalization_4_layer_call_fn_30322

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
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_265672
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
¾
í
G__inference_sequential_4_layer_call_and_return_conditional_losses_26708
conv2d_4_input
conv2d_4_26603
conv2d_4_26605
batch_normalization_4_26698
batch_normalization_4_26700
batch_normalization_4_26702
batch_normalization_4_26704
identity¢-batch_normalization_4/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¥
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCallconv2d_4_inputconv2d_4_26603conv2d_4_26605*
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_265922"
 conv2d_4/StatefulPartitionedCall
activation_5/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
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
G__inference_activation_5_layer_call_and_return_conditional_losses_266262
activation_5/PartitionedCall¹
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:0batch_normalization_4_26698batch_normalization_4_26700batch_normalization_4_26702batch_normalization_4_26704*
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
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_266532/
-batch_normalization_4/StatefulPartitionedCallæ
IdentityIdentity6batch_normalization_4/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:ÿÿÿÿÿÿÿÿÿ@@::::::2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:` \
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@
(
_user_specified_nameconv2d_4_input
·E
ú	
G__inference_functional_1_layer_call_and_return_conditional_losses_27703
input_1
sequential_27613
sequential_27615
sequential_27617
sequential_27619
sequential_27621
sequential_27623
sequential_1_27627
sequential_1_27629
sequential_1_27631
sequential_1_27633
sequential_1_27635
sequential_1_27637
sequential_2_27641
sequential_2_27643
sequential_2_27645
sequential_2_27647
sequential_2_27649
sequential_2_27651
sequential_3_27655
sequential_3_27657
sequential_3_27659
sequential_3_27661
sequential_3_27663
sequential_3_27665
sequential_4_27669
sequential_4_27671
sequential_4_27673
sequential_4_27675
sequential_4_27677
sequential_4_27679
sequential_5_27683
sequential_5_27685
sequential_5_27687
sequential_5_27689
sequential_5_27691
sequential_5_27693
sequential_6_27697
sequential_6_27699
identity¢"sequential/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCall¢$sequential_2/StatefulPartitionedCall¢$sequential_3/StatefulPartitionedCall¢$sequential_4/StatefulPartitionedCall¢$sequential_5/StatefulPartitionedCall¢$sequential_6/StatefulPartitionedCallù
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_27613sequential_27615sequential_27617sequential_27619sequential_27621sequential_27623*
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
E__inference_sequential_layer_call_and_return_conditional_losses_254252$
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
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_254462!
max_pooling2d_3/PartitionedCall¯
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall+sequential/StatefulPartitionedCall:output:0sequential_1_27627sequential_1_27629sequential_1_27631sequential_1_27633sequential_1_27635sequential_1_27637*
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_257792&
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
F__inference_concatenate_layer_call_and_return_conditional_losses_273552
concatenate/PartitionedCall¨
$sequential_2/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0sequential_2_27641sequential_2_27643sequential_2_27645sequential_2_27647sequential_2_27649sequential_2_27651*
Tin
	2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_261052&
$sequential_2/StatefulPartitionedCall
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
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_261262!
max_pooling2d_4/PartitionedCall¯
$sequential_3/StatefulPartitionedCallStatefulPartitionedCall-sequential_2/StatefulPartitionedCall:output:0sequential_3_27655sequential_3_27657sequential_3_27659sequential_3_27661sequential_3_27663sequential_3_27665*
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
G__inference_sequential_3_layer_call_and_return_conditional_losses_264592&
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
H__inference_concatenate_1_layer_call_and_return_conditional_losses_274662
concatenate_1/PartitionedCall©
$sequential_4/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0sequential_4_27669sequential_4_27671sequential_4_27673sequential_4_27675sequential_4_27677sequential_4_27679*
Tin
	2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@@*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_sequential_4_layer_call_and_return_conditional_losses_267852&
$sequential_4/StatefulPartitionedCall
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
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_268062!
max_pooling2d_5/PartitionedCall°
$sequential_5/StatefulPartitionedCallStatefulPartitionedCall-sequential_4/StatefulPartitionedCall:output:0sequential_5_27683sequential_5_27685sequential_5_27687sequential_5_27689sequential_5_27691sequential_5_27693*
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
G__inference_sequential_5_layer_call_and_return_conditional_losses_271392&
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
H__inference_concatenate_2_layer_call_and_return_conditional_losses_275772
concatenate_2/PartitionedCallÈ
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0sequential_6_27697sequential_6_27699*
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
G__inference_sequential_6_layer_call_and_return_conditional_losses_272422&
$sequential_6/StatefulPartitionedCall
IdentityIdentity-sequential_6/StatefulPartitionedCall:output:0#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall%^sequential_2/StatefulPartitionedCall%^sequential_3/StatefulPartitionedCall%^sequential_4/StatefulPartitionedCall%^sequential_5/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Ê
_input_shapes¸
µ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::::::::::::::::::::2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall2L
$sequential_2/StatefulPartitionedCall$sequential_2/StatefulPartitionedCall2L
$sequential_3/StatefulPartitionedCall$sequential_3/StatefulPartitionedCall2L
$sequential_4/StatefulPartitionedCall$sequential_4/StatefulPartitionedCall2L
$sequential_5/StatefulPartitionedCall$sequential_5/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:Z V
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1

f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_26242

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
®
«
C__inference_conv2d_3_layer_call_and_return_conditional_losses_26262

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

½
,__inference_sequential_5_layer_call_fn_29462

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
G__inference_sequential_5_layer_call_and_return_conditional_losses_271022
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
Ö
í
G__inference_sequential_1_layer_call_and_return_conditional_losses_25699
conv2d_1_input
conv2d_1_25593
conv2d_1_25595
batch_normalization_1_25688
batch_normalization_1_25690
batch_normalization_1_25692
batch_normalization_1_25694
identity¢-batch_normalization_1/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¦
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputconv2d_1_25593conv2d_1_25595*
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_255822"
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
G__inference_activation_2_layer_call_and_return_conditional_losses_256162
activation_2/PartitionedCallº
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0batch_normalization_1_25688batch_normalization_1_25690batch_normalization_1_25692batch_normalization_1_25694*
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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_256432/
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_255622
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
ç&
³
G__inference_sequential_3_layer_call_and_return_conditional_losses_29206

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
à
¨
5__inference_batch_normalization_3_layer_call_fn_30216

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
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_263412
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

»
*__inference_sequential_layer_call_fn_28887

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
E__inference_sequential_layer_call_and_return_conditional_losses_254252
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


P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_25887

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
®
K
/__inference_max_pooling2d_3_layer_call_fn_25452

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
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_254462
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
®
«
C__inference_conv2d_3_layer_call_and_return_conditional_losses_30056

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
Æ
í
G__inference_sequential_2_layer_call_and_return_conditional_losses_26047
conv2d_2_input
conv2d_2_26031
conv2d_2_26033
batch_normalization_2_26037
batch_normalization_2_26039
batch_normalization_2_26041
batch_normalization_2_26043
identity¢-batch_normalization_2/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¦
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallconv2d_2_inputconv2d_2_26031conv2d_2_26033*
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_259122"
 conv2d_2/StatefulPartitionedCall
activation_3/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
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
G__inference_activation_3_layer_call_and_return_conditional_losses_259462
activation_3/PartitionedCall¼
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:0batch_normalization_2_26037batch_normalization_2_26039batch_normalization_2_26041batch_normalization_2_26043*
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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_259912/
-batch_normalization_2/StatefulPartitionedCallç
IdentityIdentity6batch_normalization_2/StatefulPartitionedCall:output:0.^batch_normalization_2/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:ÿÿÿÿÿÿÿÿÿ@::::::2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:a ]
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
(
_user_specified_nameconv2d_2_input
Û

P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_30530

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
¾
å
G__inference_sequential_1_layer_call_and_return_conditional_losses_25742

inputs
conv2d_1_25725
conv2d_1_25727
batch_normalization_1_25731
batch_normalization_1_25733
batch_normalization_1_25735
batch_normalization_1_25737
identity¢-batch_normalization_1/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_25725conv2d_1_25727*
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_255822"
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
G__inference_activation_2_layer_call_and_return_conditional_losses_256162
activation_2/PartitionedCallº
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0batch_normalization_1_25731batch_normalization_1_25733batch_normalization_1_25735batch_normalization_1_25737*
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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_256432/
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_255622
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
ç&
³
G__inference_sequential_5_layer_call_and_return_conditional_losses_29445

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
°
c
G__inference_activation_4_layer_call_and_return_conditional_losses_26296

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
Ù
í
G__inference_sequential_5_layer_call_and_return_conditional_losses_27079
conv2d_5_input
conv2d_5_27062
conv2d_5_27064
batch_normalization_5_27068
batch_normalization_5_27070
batch_normalization_5_27072
batch_normalization_5_27074
identity¢-batch_normalization_5/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall¥
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallconv2d_5_inputconv2d_5_27062conv2d_5_27064*
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_269422"
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
G__inference_activation_6_layer_call_and_return_conditional_losses_269762
activation_6/PartitionedCall»
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0batch_normalization_5_27068batch_normalization_5_27070batch_normalization_5_27072batch_normalization_5_27074*
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
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_270212/
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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_269222!
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
_user_specified_nameconv2d_5_input"¸L
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
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¸

Àç
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer-12
layer_with_weights-6
layer-13
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+ü&call_and_return_all_conditional_losses
ý__call__
þ_default_save_signature"ã
_tf_keras_networkäâ{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["sequential", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_1", "inbound_nodes": [[["sequential", 1, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}], ["sequential_1", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "sequential_2", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_3", "inbound_nodes": [[["sequential_2", 1, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}], ["sequential_3", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "sequential_4", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_5", "inbound_nodes": [[["sequential_4", 1, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}], ["sequential_5", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_input"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 29, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_6", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["sequential_6", 1, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["sequential", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_1", "inbound_nodes": [[["sequential", 1, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}], ["sequential_1", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "sequential_2", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_3", "inbound_nodes": [[["sequential_2", 1, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}], ["sequential_3", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "name": "sequential_4", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["concatenate_1", 0, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "name": "sequential_5", "inbound_nodes": [[["sequential_4", 1, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}], ["sequential_5", 1, 0, {}]]]}, {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_input"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 29, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential_6", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["sequential_6", 1, 0]]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ý"ú
_tf_keras_input_layerÚ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
¸
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
+ÿ&call_and_return_all_conditional_losses
__call__"Ì
_tf_keras_sequential­{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}}}

	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
$
 layer_with_weights-0
 layer-0
!layer-1
"layer_with_weights-1
"layer-2
#layer-3
$	variables
%trainable_variables
&regularization_losses
'	keras_api
+&call_and_return_all_conditional_losses
__call__""
_tf_keras_sequentialë!{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 32]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 32]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_1_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}}
ß
(	variables
)trainable_variables
*regularization_losses
+	keras_api
+&call_and_return_all_conditional_losses
__call__"Î
_tf_keras_layer´{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 128, 32]}, {"class_name": "TensorShape", "items": [null, 128, 128, 32]}]}
Î
,layer_with_weights-0
,layer-0
-layer-1
.layer_with_weights-1
.layer-2
/	variables
0trainable_variables
1regularization_losses
2	keras_api
+&call_and_return_all_conditional_losses
__call__"â
_tf_keras_sequentialÃ{"class_name": "Sequential", "name": "sequential_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_2_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}}}

3	variables
4trainable_variables
5regularization_losses
6	keras_api
+&call_and_return_all_conditional_losses
__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
$
7layer_with_weights-0
7layer-0
8layer-1
9layer_with_weights-1
9layer-2
:layer-3
;	variables
<trainable_variables
=regularization_losses
>	keras_api
+&call_and_return_all_conditional_losses
__call__""
_tf_keras_sequentialï!{"class_name": "Sequential", "name": "sequential_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 128, 128, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_3_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}}
ß
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
+&call_and_return_all_conditional_losses
__call__"Î
_tf_keras_layer´{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64, 64, 64]}, {"class_name": "TensorShape", "items": [null, 64, 64, 64]}]}
Î
Clayer_with_weights-0
Clayer-0
Dlayer-1
Elayer_with_weights-1
Elayer-2
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
+&call_and_return_all_conditional_losses
__call__"â
_tf_keras_sequentialÃ{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_4_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}]}}}

J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
+&call_and_return_all_conditional_losses
__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
$
Nlayer_with_weights-0
Nlayer-0
Olayer-1
Player_with_weights-1
Player-2
Qlayer-3
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
+&call_and_return_all_conditional_losses
__call__""
_tf_keras_sequentialï!{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_5_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}]}}}
á
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
+&call_and_return_all_conditional_losses
__call__"Ð
_tf_keras_layer¶{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32, 32, 128]}, {"class_name": "TensorShape", "items": [null, 32, 32, 128]}]}
¾
Zlayer-0
[layer_with_weights-0
[layer-1
\	variables
]trainable_variables
^regularization_losses
_	keras_api
+&call_and_return_all_conditional_losses
__call__"ù
_tf_keras_sequentialÚ{"class_name": "Sequential", "name": "sequential_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_input"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 29, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 256]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 256]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_input"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 29, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
é
`iter

abeta_1

bbeta_2
	cdecay
dlearning_rateemÈfmÉgmÊhmËkmÌlmÍmmÎnmÏqmÐrmÑsmÒtmÓwmÔxmÕymÖzm×}mØ~mÙmÚ	mÛ	mÜ	mÝ	mÞ	mß	mà	máevâfvãgvähvåkvælvçmvènvéqvêrvësvìtvíwvîxvïyvðzvñ}vò~vóvô	võ	vö	v÷	vø	vù	vú	vû"
	optimizer
Ñ
e0
f1
g2
h3
i4
j5
k6
l7
m8
n9
o10
p11
q12
r13
s14
t15
u16
v17
w18
x19
y20
z21
{22
|23
}24
~25
26
27
28
29
30
31
32
33
34
35
36
37"
trackable_list_wrapper
í
e0
f1
g2
h3
k4
l5
m6
n7
q8
r9
s10
t11
w12
x13
y14
z15
}16
~17
18
19
20
21
22
23
24
25"
trackable_list_wrapper
 "
trackable_list_wrapper
Ó
layers
 layer_regularization_losses
	variables
layer_metrics
trainable_variables
regularization_losses
non_trainable_variables
metrics
ý__call__
þ_default_save_signature
+ü&call_and_return_all_conditional_losses
'ü"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map


_inbound_nodes

ekernel
fbias
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Ë
_tf_keras_layer±{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}}
÷
_inbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"Í
_tf_keras_layer³{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}
Ô	
_inbound_nodes
	axis
	ggamma
hbeta
imoving_mean
jmoving_variance
	variables
trainable_variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"ä
_tf_keras_layerÊ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 32]}}
J
e0
f1
g2
h3
i4
j5"
trackable_list_wrapper
<
e0
f1
g2
h3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 layers
 ¡layer_regularization_losses
	variables
¢layer_metrics
trainable_variables
regularization_losses
£non_trainable_variables
¤metrics
__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¥layers
 ¦layer_regularization_losses
§layer_metrics
	variables
trainable_variables
regularization_losses
¨non_trainable_variables
©metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


ª_inbound_nodes

kkernel
lbias
«	variables
¬trainable_variables
­regularization_losses
®	keras_api
+ &call_and_return_all_conditional_losses
¡__call__"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 32]}}
÷
¯_inbound_nodes
°	variables
±trainable_variables
²regularization_losses
³	keras_api
+¢&call_and_return_all_conditional_losses
£__call__"Í
_tf_keras_layer³{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}
Ø	
´_inbound_nodes
	µaxis
	mgamma
nbeta
omoving_mean
pmoving_variance
¶	variables
·trainable_variables
¸regularization_losses
¹	keras_api
+¤&call_and_return_all_conditional_losses
¥__call__"è
_tf_keras_layerÎ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 32]}}

º_inbound_nodes
»	variables
¼trainable_variables
½regularization_losses
¾	keras_api
+¦&call_and_return_all_conditional_losses
§__call__"ì
_tf_keras_layerÒ{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
J
k0
l1
m2
n3
o4
p5"
trackable_list_wrapper
<
k0
l1
m2
n3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¿layers
 Àlayer_regularization_losses
$	variables
Álayer_metrics
%trainable_variables
&regularization_losses
Ânon_trainable_variables
Ãmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Älayers
 Ålayer_regularization_losses
Ælayer_metrics
(	variables
)trainable_variables
*regularization_losses
Çnon_trainable_variables
Èmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


É_inbound_nodes

qkernel
rbias
Ê	variables
Ëtrainable_variables
Ìregularization_losses
Í	keras_api
+¨&call_and_return_all_conditional_losses
©__call__"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 64]}}
÷
Î_inbound_nodes
Ï	variables
Ðtrainable_variables
Ñregularization_losses
Ò	keras_api
+ª&call_and_return_all_conditional_losses
«__call__"Í
_tf_keras_layer³{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}
Ø	
Ó_inbound_nodes
	Ôaxis
	sgamma
tbeta
umoving_mean
vmoving_variance
Õ	variables
Ötrainable_variables
×regularization_losses
Ø	keras_api
+¬&call_and_return_all_conditional_losses
­__call__"è
_tf_keras_layerÎ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 64]}}
J
q0
r1
s2
t3
u4
v5"
trackable_list_wrapper
<
q0
r1
s2
t3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ùlayers
 Úlayer_regularization_losses
/	variables
Ûlayer_metrics
0trainable_variables
1regularization_losses
Ünon_trainable_variables
Ýmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Þlayers
 ßlayer_regularization_losses
àlayer_metrics
3	variables
4trainable_variables
5regularization_losses
ánon_trainable_variables
âmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


ã_inbound_nodes

wkernel
xbias
ä	variables
åtrainable_variables
æregularization_losses
ç	keras_api
+®&call_and_return_all_conditional_losses
¯__call__"Ñ
_tf_keras_layer·{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 64]}}
÷
è_inbound_nodes
é	variables
êtrainable_variables
ëregularization_losses
ì	keras_api
+°&call_and_return_all_conditional_losses
±__call__"Í
_tf_keras_layer³{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}
Ø	
í_inbound_nodes
	îaxis
	ygamma
zbeta
{moving_mean
|moving_variance
ï	variables
ðtrainable_variables
ñregularization_losses
ò	keras_api
+²&call_and_return_all_conditional_losses
³__call__"è
_tf_keras_layerÎ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 64]}}

ó_inbound_nodes
ô	variables
õtrainable_variables
öregularization_losses
÷	keras_api
+´&call_and_return_all_conditional_losses
µ__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
J
w0
x1
y2
z3
{4
|5"
trackable_list_wrapper
<
w0
x1
y2
z3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ølayers
 ùlayer_regularization_losses
;	variables
úlayer_metrics
<trainable_variables
=regularization_losses
ûnon_trainable_variables
ümetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ýlayers
 þlayer_regularization_losses
ÿlayer_metrics
?	variables
@trainable_variables
Aregularization_losses
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


_inbound_nodes

}kernel
~bias
	variables
trainable_variables
regularization_losses
	keras_api
+¶&call_and_return_all_conditional_losses
·__call__"Ò
_tf_keras_layer¸{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 128]}}
÷
_inbound_nodes
	variables
trainable_variables
regularization_losses
	keras_api
+¸&call_and_return_all_conditional_losses
¹__call__"Í
_tf_keras_layer³{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}
Û	
_inbound_nodes
	axis
	gamma
	beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
+º&call_and_return_all_conditional_losses
»__call__"è
_tf_keras_layerÎ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 128]}}
M
}0
~1
2
3
4
5"
trackable_list_wrapper
=
}0
~1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
 layer_regularization_losses
F	variables
layer_metrics
Gtrainable_variables
Hregularization_losses
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
 layer_regularization_losses
layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
non_trainable_variables
metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object


_inbound_nodes
kernel
	bias
	variables
trainable_variables
regularization_losses
 	keras_api
+¼&call_and_return_all_conditional_losses
½__call__"Ò
_tf_keras_layer¸{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 128]}}
÷
¡_inbound_nodes
¢	variables
£trainable_variables
¤regularization_losses
¥	keras_api
+¾&call_and_return_all_conditional_losses
¿__call__"Í
_tf_keras_layer³{"class_name": "Activation", "name": "activation_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "custom_gelu"}}
Ü	
¦_inbound_nodes
	§axis

gamma
	beta
moving_mean
moving_variance
¨	variables
©trainable_variables
ªregularization_losses
«	keras_api
+À&call_and_return_all_conditional_losses
Á__call__"è
_tf_keras_layerÎ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 128]}}

¬_inbound_nodes
­	variables
®trainable_variables
¯regularization_losses
°	keras_api
+Â&call_and_return_all_conditional_losses
Ã__call__"ð
_tf_keras_layerÖ{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
P
0
1
2
3
4
5"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
±layers
 ²layer_regularization_losses
R	variables
³layer_metrics
Strainable_variables
Tregularization_losses
´non_trainable_variables
µmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¶layers
 ·layer_regularization_losses
¸layer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
¹non_trainable_variables
ºmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ý
»_inbound_nodes
¼	variables
½trainable_variables
¾regularization_losses
¿	keras_api
+Ä&call_and_return_all_conditional_losses
Å__call__"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}

À_inbound_nodes
kernel
	bias
Á	variables
Âtrainable_variables
Ãregularization_losses
Ä	keras_api
+Æ&call_and_return_all_conditional_losses
Ç__call__"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 29, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 262144}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 262144]}}
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ålayers
 Ælayer_regularization_losses
\	variables
Çlayer_metrics
]trainable_variables
^regularization_losses
Ènon_trainable_variables
Émetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
::8  (2*sequential/batch_normalization/moving_mean
>:<  (2.sequential/batch_normalization/moving_variance
6:4  2sequential_1/conv2d_1/kernel
(:& 2sequential_1/conv2d_1/bias
6:4 2(sequential_1/batch_normalization_1/gamma
5:3 2'sequential_1/batch_normalization_1/beta
>:<  (2.sequential_1/batch_normalization_1/moving_mean
B:@  (22sequential_1/batch_normalization_1/moving_variance
6:4@@2sequential_2/conv2d_2/kernel
(:&@2sequential_2/conv2d_2/bias
6:4@2(sequential_2/batch_normalization_2/gamma
5:3@2'sequential_2/batch_normalization_2/beta
>:<@ (2.sequential_2/batch_normalization_2/moving_mean
B:@@ (22sequential_2/batch_normalization_2/moving_variance
6:4@@2sequential_3/conv2d_3/kernel
(:&@2sequential_3/conv2d_3/bias
6:4@2(sequential_3/batch_normalization_3/gamma
5:3@2'sequential_3/batch_normalization_3/beta
>:<@ (2.sequential_3/batch_normalization_3/moving_mean
B:@@ (22sequential_3/batch_normalization_3/moving_variance
8:62sequential_4/conv2d_4/kernel
):'2sequential_4/conv2d_4/bias
7:52(sequential_4/batch_normalization_4/gamma
6:42'sequential_4/batch_normalization_4/beta
?:= (2.sequential_4/batch_normalization_4/moving_mean
C:A (22sequential_4/batch_normalization_4/moving_variance
8:62sequential_5/conv2d_5/kernel
):'2sequential_5/conv2d_5/bias
7:52(sequential_5/batch_normalization_5/gamma
6:42'sequential_5/batch_normalization_5/beta
?:= (2.sequential_5/batch_normalization_5/moving_mean
C:A (22sequential_5/batch_normalization_5/moving_variance
-:+
2sequential_6/dense/kernel
%:#2sequential_6/dense/bias

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
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
z
i0
j1
o2
p3
u4
v5
{6
|7
8
9
10
11"
trackable_list_wrapper
0
Ê0
Ë1"
trackable_list_wrapper
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
Ìlayers
 Ílayer_regularization_losses
Îlayer_metrics
	variables
trainable_variables
regularization_losses
Ïnon_trainable_variables
Ðmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
Ñlayers
 Òlayer_regularization_losses
Ólayer_metrics
	variables
trainable_variables
regularization_losses
Ônon_trainable_variables
Õmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
g0
h1
i2
j3"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ölayers
 ×layer_regularization_losses
Ølayer_metrics
	variables
trainable_variables
regularization_losses
Ùnon_trainable_variables
Úmetrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
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
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ûlayers
 Ülayer_regularization_losses
Ýlayer_metrics
«	variables
¬trainable_variables
­regularization_losses
Þnon_trainable_variables
ßmetrics
¡__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
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
àlayers
 álayer_regularization_losses
âlayer_metrics
°	variables
±trainable_variables
²regularization_losses
ãnon_trainable_variables
ämetrics
£__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
m0
n1
o2
p3"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ålayers
 ælayer_regularization_losses
çlayer_metrics
¶	variables
·trainable_variables
¸regularization_losses
ènon_trainable_variables
émetrics
¥__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
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
êlayers
 ëlayer_regularization_losses
ìlayer_metrics
»	variables
¼trainable_variables
½regularization_losses
ínon_trainable_variables
îmetrics
§__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
<
 0
!1
"2
#3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
o0
p1"
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
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ïlayers
 ðlayer_regularization_losses
ñlayer_metrics
Ê	variables
Ëtrainable_variables
Ìregularization_losses
ònon_trainable_variables
ómetrics
©__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
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
ôlayers
 õlayer_regularization_losses
ölayer_metrics
Ï	variables
Ðtrainable_variables
Ñregularization_losses
÷non_trainable_variables
ømetrics
«__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
s0
t1
u2
v3"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ùlayers
 úlayer_regularization_losses
ûlayer_metrics
Õ	variables
Ötrainable_variables
×regularization_losses
ünon_trainable_variables
ýmetrics
­__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
5
,0
-1
.2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
u0
v1"
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
w0
x1"
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
þlayers
 ÿlayer_regularization_losses
layer_metrics
ä	variables
åtrainable_variables
æregularization_losses
non_trainable_variables
metrics
¯__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
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
layers
 layer_regularization_losses
layer_metrics
é	variables
êtrainable_variables
ëregularization_losses
non_trainable_variables
metrics
±__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
y0
z1
{2
|3"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
 layer_regularization_losses
layer_metrics
ï	variables
ðtrainable_variables
ñregularization_losses
non_trainable_variables
metrics
³__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
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
layers
 layer_regularization_losses
layer_metrics
ô	variables
õtrainable_variables
öregularization_losses
non_trainable_variables
metrics
µ__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
<
70
81
92
:3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
{0
|1"
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
}0
~1"
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
non_trainable_variables
metrics
·__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
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
layers
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
non_trainable_variables
metrics
¹__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3"
trackable_list_wrapper
/
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
layers
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
non_trainable_variables
 metrics
»__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
5
C0
D1
E2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
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
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¡layers
 ¢layer_regularization_losses
£layer_metrics
	variables
trainable_variables
regularization_losses
¤non_trainable_variables
¥metrics
½__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
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
¦layers
 §layer_regularization_losses
¨layer_metrics
¢	variables
£trainable_variables
¤regularization_losses
©non_trainable_variables
ªmetrics
¿__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
«layers
 ¬layer_regularization_losses
­layer_metrics
¨	variables
©trainable_variables
ªregularization_losses
®non_trainable_variables
¯metrics
Á__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
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
°layers
 ±layer_regularization_losses
²layer_metrics
­	variables
®trainable_variables
¯regularization_losses
³non_trainable_variables
´metrics
Ã__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
<
N0
O1
P2
Q3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
0
1"
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
µlayers
 ¶layer_regularization_losses
·layer_metrics
¼	variables
½trainable_variables
¾regularization_losses
¸non_trainable_variables
¹metrics
Å__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ºlayers
 »layer_regularization_losses
¼layer_metrics
Á	variables
Âtrainable_variables
Ãregularization_losses
½non_trainable_variables
¾metrics
Ç__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿

¿total

Àcount
Á	variables
Â	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}


Ãtotal

Äcount
Å
_fn_kwargs
Æ	variables
Ç	keras_api"¸
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
.
o0
p1"
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
u0
v1"
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
{0
|1"
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
0
0
1"
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
0
0
1"
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
¿0
À1"
trackable_list_wrapper
.
Á	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ã0
Ä1"
trackable_list_wrapper
.
Æ	variables"
_generic_user_object
7:5 2Adam/sequential/conv2d/kernel/m
):' 2Adam/sequential/conv2d/bias/m
7:5 2+Adam/sequential/batch_normalization/gamma/m
6:4 2*Adam/sequential/batch_normalization/beta/m
;:9  2#Adam/sequential_1/conv2d_1/kernel/m
-:+ 2!Adam/sequential_1/conv2d_1/bias/m
;:9 2/Adam/sequential_1/batch_normalization_1/gamma/m
::8 2.Adam/sequential_1/batch_normalization_1/beta/m
;:9@@2#Adam/sequential_2/conv2d_2/kernel/m
-:+@2!Adam/sequential_2/conv2d_2/bias/m
;:9@2/Adam/sequential_2/batch_normalization_2/gamma/m
::8@2.Adam/sequential_2/batch_normalization_2/beta/m
;:9@@2#Adam/sequential_3/conv2d_3/kernel/m
-:+@2!Adam/sequential_3/conv2d_3/bias/m
;:9@2/Adam/sequential_3/batch_normalization_3/gamma/m
::8@2.Adam/sequential_3/batch_normalization_3/beta/m
=:;2#Adam/sequential_4/conv2d_4/kernel/m
.:,2!Adam/sequential_4/conv2d_4/bias/m
<::2/Adam/sequential_4/batch_normalization_4/gamma/m
;:92.Adam/sequential_4/batch_normalization_4/beta/m
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
;:9@@2#Adam/sequential_2/conv2d_2/kernel/v
-:+@2!Adam/sequential_2/conv2d_2/bias/v
;:9@2/Adam/sequential_2/batch_normalization_2/gamma/v
::8@2.Adam/sequential_2/batch_normalization_2/beta/v
;:9@@2#Adam/sequential_3/conv2d_3/kernel/v
-:+@2!Adam/sequential_3/conv2d_3/bias/v
;:9@2/Adam/sequential_3/batch_normalization_3/gamma/v
::8@2.Adam/sequential_3/batch_normalization_3/beta/v
=:;2#Adam/sequential_4/conv2d_4/kernel/v
.:,2!Adam/sequential_4/conv2d_4/bias/v
<::2/Adam/sequential_4/batch_normalization_4/gamma/v
;:92.Adam/sequential_4/batch_normalization_4/beta/v
=:;2#Adam/sequential_5/conv2d_5/kernel/v
.:,2!Adam/sequential_5/conv2d_5/bias/v
<::2/Adam/sequential_5/batch_normalization_5/gamma/v
;:92.Adam/sequential_5/batch_normalization_5/beta/v
2:0
2 Adam/sequential_6/dense/kernel/v
*:(2Adam/sequential_6/dense/bias/v
ê2ç
G__inference_functional_1_layer_call_and_return_conditional_losses_28384
G__inference_functional_1_layer_call_and_return_conditional_losses_28613
G__inference_functional_1_layer_call_and_return_conditional_losses_27610
G__inference_functional_1_layer_call_and_return_conditional_losses_27703À
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
þ2û
,__inference_functional_1_layer_call_fn_27878
,__inference_functional_1_layer_call_fn_28775
,__inference_functional_1_layer_call_fn_28052
,__inference_functional_1_layer_call_fn_28694À
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
è2å
 __inference__wrapped_model_25114À
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
â2ß
E__inference_sequential_layer_call_and_return_conditional_losses_25348
E__inference_sequential_layer_call_and_return_conditional_losses_28853
E__inference_sequential_layer_call_and_return_conditional_losses_28815
E__inference_sequential_layer_call_and_return_conditional_losses_25367À
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
*__inference_sequential_layer_call_fn_25440
*__inference_sequential_layer_call_fn_28887
*__inference_sequential_layer_call_fn_25404
*__inference_sequential_layer_call_fn_28870À
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
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_25446à
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
/__inference_max_pooling2d_3_layer_call_fn_25452à
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
ê2ç
G__inference_sequential_1_layer_call_and_return_conditional_losses_25719
G__inference_sequential_1_layer_call_and_return_conditional_losses_28967
G__inference_sequential_1_layer_call_and_return_conditional_losses_28928
G__inference_sequential_1_layer_call_and_return_conditional_losses_25699À
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
þ2û
,__inference_sequential_1_layer_call_fn_29001
,__inference_sequential_1_layer_call_fn_28984
,__inference_sequential_1_layer_call_fn_25757
,__inference_sequential_1_layer_call_fn_25794À
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
F__inference_concatenate_layer_call_and_return_conditional_losses_29008¢
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
+__inference_concatenate_layer_call_fn_29014¢
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
ê2ç
G__inference_sequential_2_layer_call_and_return_conditional_losses_26047
G__inference_sequential_2_layer_call_and_return_conditional_losses_29054
G__inference_sequential_2_layer_call_and_return_conditional_losses_29092
G__inference_sequential_2_layer_call_and_return_conditional_losses_26028À
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
þ2û
,__inference_sequential_2_layer_call_fn_29109
,__inference_sequential_2_layer_call_fn_26084
,__inference_sequential_2_layer_call_fn_29126
,__inference_sequential_2_layer_call_fn_26120À
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
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_26126à
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
/__inference_max_pooling2d_4_layer_call_fn_26132à
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
ê2ç
G__inference_sequential_3_layer_call_and_return_conditional_losses_29206
G__inference_sequential_3_layer_call_and_return_conditional_losses_29167
G__inference_sequential_3_layer_call_and_return_conditional_losses_26379
G__inference_sequential_3_layer_call_and_return_conditional_losses_26399À
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
þ2û
,__inference_sequential_3_layer_call_fn_26437
,__inference_sequential_3_layer_call_fn_29240
,__inference_sequential_3_layer_call_fn_29223
,__inference_sequential_3_layer_call_fn_26474À
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
H__inference_concatenate_1_layer_call_and_return_conditional_losses_29247¢
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
-__inference_concatenate_1_layer_call_fn_29253¢
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
ê2ç
G__inference_sequential_4_layer_call_and_return_conditional_losses_29293
G__inference_sequential_4_layer_call_and_return_conditional_losses_26708
G__inference_sequential_4_layer_call_and_return_conditional_losses_29331
G__inference_sequential_4_layer_call_and_return_conditional_losses_26727À
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
þ2û
,__inference_sequential_4_layer_call_fn_29365
,__inference_sequential_4_layer_call_fn_29348
,__inference_sequential_4_layer_call_fn_26764
,__inference_sequential_4_layer_call_fn_26800À
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
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_26806à
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
/__inference_max_pooling2d_5_layer_call_fn_26812à
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
ê2ç
G__inference_sequential_5_layer_call_and_return_conditional_losses_27079
G__inference_sequential_5_layer_call_and_return_conditional_losses_29406
G__inference_sequential_5_layer_call_and_return_conditional_losses_29445
G__inference_sequential_5_layer_call_and_return_conditional_losses_27059À
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
þ2û
,__inference_sequential_5_layer_call_fn_29462
,__inference_sequential_5_layer_call_fn_27154
,__inference_sequential_5_layer_call_fn_29479
,__inference_sequential_5_layer_call_fn_27117À
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
H__inference_concatenate_2_layer_call_and_return_conditional_losses_29486¢
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
-__inference_concatenate_2_layer_call_fn_29492¢
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
ê2ç
G__inference_sequential_6_layer_call_and_return_conditional_losses_27200
G__inference_sequential_6_layer_call_and_return_conditional_losses_29505
G__inference_sequential_6_layer_call_and_return_conditional_losses_29518
G__inference_sequential_6_layer_call_and_return_conditional_losses_27210À
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
þ2û
,__inference_sequential_6_layer_call_fn_29527
,__inference_sequential_6_layer_call_fn_29536
,__inference_sequential_6_layer_call_fn_27230
,__inference_sequential_6_layer_call_fn_27249À
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
#__inference_signature_wrapper_28143input_1
ë2è
A__inference_conv2d_layer_call_and_return_conditional_losses_29546¢
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
&__inference_conv2d_layer_call_fn_29555¢
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
G__inference_activation_1_layer_call_and_return_conditional_losses_29573¢
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
,__inference_activation_1_layer_call_fn_29578¢
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
N__inference_batch_normalization_layer_call_and_return_conditional_losses_29616
N__inference_batch_normalization_layer_call_and_return_conditional_losses_29680
N__inference_batch_normalization_layer_call_and_return_conditional_losses_29598
N__inference_batch_normalization_layer_call_and_return_conditional_losses_29662´
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
3__inference_batch_normalization_layer_call_fn_29642
3__inference_batch_normalization_layer_call_fn_29706
3__inference_batch_normalization_layer_call_fn_29693
3__inference_batch_normalization_layer_call_fn_29629´
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_29716¢
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
(__inference_conv2d_1_layer_call_fn_29725¢
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
G__inference_activation_2_layer_call_and_return_conditional_losses_29743¢
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
,__inference_activation_2_layer_call_fn_29748¢
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
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29786
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29768
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29832
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29850´
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
5__inference_batch_normalization_1_layer_call_fn_29812
5__inference_batch_normalization_1_layer_call_fn_29876
5__inference_batch_normalization_1_layer_call_fn_29799
5__inference_batch_normalization_1_layer_call_fn_29863´
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
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25562à
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
-__inference_max_pooling2d_layer_call_fn_25568à
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_29886¢
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
(__inference_conv2d_2_layer_call_fn_29895¢
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
G__inference_activation_3_layer_call_and_return_conditional_losses_29913¢
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
,__inference_activation_3_layer_call_fn_29918¢
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
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_29938
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_30020
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_30002
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_29956´
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
5__inference_batch_normalization_2_layer_call_fn_29982
5__inference_batch_normalization_2_layer_call_fn_30046
5__inference_batch_normalization_2_layer_call_fn_29969
5__inference_batch_normalization_2_layer_call_fn_30033´
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
C__inference_conv2d_3_layer_call_and_return_conditional_losses_30056¢
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
(__inference_conv2d_3_layer_call_fn_30065¢
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
G__inference_activation_4_layer_call_and_return_conditional_losses_30083¢
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
,__inference_activation_4_layer_call_fn_30088¢
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
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_30126
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_30108
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_30172
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_30190´
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
5__inference_batch_normalization_3_layer_call_fn_30216
5__inference_batch_normalization_3_layer_call_fn_30152
5__inference_batch_normalization_3_layer_call_fn_30139
5__inference_batch_normalization_3_layer_call_fn_30203´
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
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_26242à
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
/__inference_max_pooling2d_1_layer_call_fn_26248à
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
C__inference_conv2d_4_layer_call_and_return_conditional_losses_30226¢
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
(__inference_conv2d_4_layer_call_fn_30235¢
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
G__inference_activation_5_layer_call_and_return_conditional_losses_30253¢
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
,__inference_activation_5_layer_call_fn_30258¢
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
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_30296
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_30342
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_30278
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_30360´
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
5__inference_batch_normalization_4_layer_call_fn_30386
5__inference_batch_normalization_4_layer_call_fn_30309
5__inference_batch_normalization_4_layer_call_fn_30373
5__inference_batch_normalization_4_layer_call_fn_30322´
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
C__inference_conv2d_5_layer_call_and_return_conditional_losses_30396¢
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
(__inference_conv2d_5_layer_call_fn_30405¢
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
G__inference_activation_6_layer_call_and_return_conditional_losses_30423¢
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
,__inference_activation_6_layer_call_fn_30428¢
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
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_30466
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_30530
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_30512
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_30448´
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
5__inference_batch_normalization_5_layer_call_fn_30556
5__inference_batch_normalization_5_layer_call_fn_30543
5__inference_batch_normalization_5_layer_call_fn_30479
5__inference_batch_normalization_5_layer_call_fn_30492´
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
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_26922à
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
/__inference_max_pooling2d_2_layer_call_fn_26928à
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
B__inference_flatten_layer_call_and_return_conditional_losses_30562¢
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
'__inference_flatten_layer_call_fn_30567¢
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
@__inference_dense_layer_call_and_return_conditional_losses_30578¢
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
%__inference_dense_layer_call_fn_30587¢
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
 Ñ
 __inference__wrapped_model_25114¬1efghijklmnopqrstuvwxyz{|}~:¢7
0¢-
+(
input_1ÿÿÿÿÿÿÿÿÿ
ª ";ª8
6
sequential_6&#
sequential_6ÿÿÿÿÿÿÿÿÿ·
G__inference_activation_1_layer_call_and_return_conditional_losses_29573l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_activation_1_layer_call_fn_29578_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿ ·
G__inference_activation_2_layer_call_and_return_conditional_losses_29743l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
,__inference_activation_2_layer_call_fn_29748_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿ ·
G__inference_activation_3_layer_call_and_return_conditional_losses_29913l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_activation_3_layer_call_fn_29918_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@·
G__inference_activation_4_layer_call_and_return_conditional_losses_30083l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
,__inference_activation_4_layer_call_fn_30088_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@µ
G__inference_activation_5_layer_call_and_return_conditional_losses_30253j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
,__inference_activation_5_layer_call_fn_30258]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@µ
G__inference_activation_6_layer_call_and_return_conditional_losses_30423j8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
,__inference_activation_6_layer_call_fn_30428]8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@ë
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29768mnopM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 ë
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29786mnopM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 Ê
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29832vmnop=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Ê
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_29850vmnop=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Ã
5__inference_batch_normalization_1_layer_call_fn_29799mnopM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Ã
5__inference_batch_normalization_1_layer_call_fn_29812mnopM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ ¢
5__inference_batch_normalization_1_layer_call_fn_29863imnop=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p
ª ""ÿÿÿÿÿÿÿÿÿ ¢
5__inference_batch_normalization_1_layer_call_fn_29876imnop=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª ""ÿÿÿÿÿÿÿÿÿ Ê
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_29938vstuv=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 Ê
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_29956vstuv=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 ë
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_30002stuvM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ë
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_30020stuvM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ¢
5__inference_batch_normalization_2_layer_call_fn_29969istuv=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ@
p
ª ""ÿÿÿÿÿÿÿÿÿ@¢
5__inference_batch_normalization_2_layer_call_fn_29982istuv=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª ""ÿÿÿÿÿÿÿÿÿ@Ã
5__inference_batch_normalization_2_layer_call_fn_30033stuvM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ã
5__inference_batch_normalization_2_layer_call_fn_30046stuvM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ë
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_30108yz{|M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 ë
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_30126yz{|M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Ê
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_30172vyz{|=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 Ê
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_30190vyz{|=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 Ã
5__inference_batch_normalization_3_layer_call_fn_30139yz{|M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ã
5__inference_batch_normalization_3_layer_call_fn_30152yz{|M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¢
5__inference_batch_normalization_3_layer_call_fn_30203iyz{|=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ@
p
ª ""ÿÿÿÿÿÿÿÿÿ@¢
5__inference_batch_normalization_3_layer_call_fn_30216iyz{|=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª ""ÿÿÿÿÿÿÿÿÿ@ð
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_30278N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ð
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_30296N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ë
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_30342w<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 Ë
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_30360w<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 È
5__inference_batch_normalization_4_layer_call_fn_30309N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
5__inference_batch_normalization_4_layer_call_fn_30322N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ£
5__inference_batch_normalization_4_layer_call_fn_30373j<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p
ª "!ÿÿÿÿÿÿÿÿÿ@@£
5__inference_batch_normalization_4_layer_call_fn_30386j<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p 
ª "!ÿÿÿÿÿÿÿÿÿ@@ñ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_30448N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ñ
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_30466N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ì
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_30512x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 Ì
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_30530x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 É
5__inference_batch_normalization_5_layer_call_fn_30479N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÉ
5__inference_batch_normalization_5_layer_call_fn_30492N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¤
5__inference_batch_normalization_5_layer_call_fn_30543k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p
ª "!ÿÿÿÿÿÿÿÿÿ@@¤
5__inference_batch_normalization_5_layer_call_fn_30556k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p 
ª "!ÿÿÿÿÿÿÿÿÿ@@é
N__inference_batch_normalization_layer_call_and_return_conditional_losses_29598ghijM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 é
N__inference_batch_normalization_layer_call_and_return_conditional_losses_29616ghijM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
 È
N__inference_batch_normalization_layer_call_and_return_conditional_losses_29662vghij=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 È
N__inference_batch_normalization_layer_call_and_return_conditional_losses_29680vghij=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 Á
3__inference_batch_normalization_layer_call_fn_29629ghijM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ Á
3__inference_batch_normalization_layer_call_fn_29642ghijM¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ 
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ  
3__inference_batch_normalization_layer_call_fn_29693ighij=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p
ª ""ÿÿÿÿÿÿÿÿÿ  
3__inference_batch_normalization_layer_call_fn_29706ighij=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 
ª ""ÿÿÿÿÿÿÿÿÿ é
H__inference_concatenate_1_layer_call_and_return_conditional_losses_29247j¢g
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
-__inference_concatenate_1_layer_call_fn_29253j¢g
`¢]
[X
*'
inputs/0ÿÿÿÿÿÿÿÿÿ@@@
*'
inputs/1ÿÿÿÿÿÿÿÿÿ@@@
ª "!ÿÿÿÿÿÿÿÿÿ@@ë
H__inference_concatenate_2_layer_call_and_return_conditional_losses_29486l¢i
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
-__inference_concatenate_2_layer_call_fn_29492l¢i
b¢_
]Z
+(
inputs/0ÿÿÿÿÿÿÿÿÿ  
+(
inputs/1ÿÿÿÿÿÿÿÿÿ  
ª "!ÿÿÿÿÿÿÿÿÿ  ì
F__inference_concatenate_layer_call_and_return_conditional_losses_29008¡n¢k
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
+__inference_concatenate_layer_call_fn_29014n¢k
d¢a
_\
,)
inputs/0ÿÿÿÿÿÿÿÿÿ 
,)
inputs/1ÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿ@·
C__inference_conv2d_1_layer_call_and_return_conditional_losses_29716pkl9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
(__inference_conv2d_1_layer_call_fn_29725ckl9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ 
ª ""ÿÿÿÿÿÿÿÿÿ ·
C__inference_conv2d_2_layer_call_and_return_conditional_losses_29886pqr9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
(__inference_conv2d_2_layer_call_fn_29895cqr9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@·
C__inference_conv2d_3_layer_call_and_return_conditional_losses_30056pwx9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 
(__inference_conv2d_3_layer_call_fn_30065cwx9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ@
ª ""ÿÿÿÿÿÿÿÿÿ@µ
C__inference_conv2d_4_layer_call_and_return_conditional_losses_30226n}~8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
(__inference_conv2d_4_layer_call_fn_30235a}~8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@·
C__inference_conv2d_5_layer_call_and_return_conditional_losses_30396p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 
(__inference_conv2d_5_layer_call_fn_30405c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ@@
ª "!ÿÿÿÿÿÿÿÿÿ@@µ
A__inference_conv2d_layer_call_and_return_conditional_losses_29546pef9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ 
 
&__inference_conv2d_layer_call_fn_29555cef9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿ ¤
@__inference_dense_layer_call_and_return_conditional_losses_30578`1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
%__inference_dense_layer_call_fn_30587S1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
B__inference_flatten_layer_call_and_return_conditional_losses_30562c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª "'¢$

0ÿÿÿÿÿÿÿÿÿ
 
'__inference_flatten_layer_call_fn_30567V8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ  
ª "ÿÿÿÿÿÿÿÿÿê
G__inference_functional_1_layer_call_and_return_conditional_losses_276101efghijklmnopqrstuvwxyz{|}~B¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ê
G__inference_functional_1_layer_call_and_return_conditional_losses_277031efghijklmnopqrstuvwxyz{|}~B¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 é
G__inference_functional_1_layer_call_and_return_conditional_losses_283841efghijklmnopqrstuvwxyz{|}~A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 é
G__inference_functional_1_layer_call_and_return_conditional_losses_286131efghijklmnopqrstuvwxyz{|}~A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
,__inference_functional_1_layer_call_fn_278781efghijklmnopqrstuvwxyz{|}~B¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÂ
,__inference_functional_1_layer_call_fn_280521efghijklmnopqrstuvwxyz{|}~B¢?
8¢5
+(
input_1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÁ
,__inference_functional_1_layer_call_fn_286941efghijklmnopqrstuvwxyz{|}~A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÁ
,__inference_functional_1_layer_call_fn_287751efghijklmnopqrstuvwxyz{|}~A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_26242R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_1_layer_call_fn_26248R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_26922R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_2_layer_call_fn_26928R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_25446R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_3_layer_call_fn_25452R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_26126R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_4_layer_call_fn_26132R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿí
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_26806R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Å
/__inference_max_pooling2d_5_layer_call_fn_26812R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿë
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_25562R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
-__inference_max_pooling2d_layer_call_fn_25568R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÐ
G__inference_sequential_1_layer_call_and_return_conditional_losses_25699klmnopI¢F
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_25719klmnopI¢F
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_28928|klmnopA¢>
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
G__inference_sequential_1_layer_call_and_return_conditional_losses_28967|klmnopA¢>
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
,__inference_sequential_1_layer_call_fn_25757wklmnopI¢F
?¢<
2/
conv2d_1_inputÿÿÿÿÿÿÿÿÿ 
p

 
ª ""ÿÿÿÿÿÿÿÿÿ §
,__inference_sequential_1_layer_call_fn_25794wklmnopI¢F
?¢<
2/
conv2d_1_inputÿÿÿÿÿÿÿÿÿ 
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ 
,__inference_sequential_1_layer_call_fn_28984oklmnopA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ 
p

 
ª ""ÿÿÿÿÿÿÿÿÿ 
,__inference_sequential_1_layer_call_fn_29001oklmnopA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ 
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ Ð
G__inference_sequential_2_layer_call_and_return_conditional_losses_26028qrstuvI¢F
?¢<
2/
conv2d_2_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 Ð
G__inference_sequential_2_layer_call_and_return_conditional_losses_26047qrstuvI¢F
?¢<
2/
conv2d_2_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 Ç
G__inference_sequential_2_layer_call_and_return_conditional_losses_29054|qrstuvA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 Ç
G__inference_sequential_2_layer_call_and_return_conditional_losses_29092|qrstuvA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ@
 §
,__inference_sequential_2_layer_call_fn_26084wqrstuvI¢F
?¢<
2/
conv2d_2_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª ""ÿÿÿÿÿÿÿÿÿ@§
,__inference_sequential_2_layer_call_fn_26120wqrstuvI¢F
?¢<
2/
conv2d_2_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ@
,__inference_sequential_2_layer_call_fn_29109oqrstuvA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª ""ÿÿÿÿÿÿÿÿÿ@
,__inference_sequential_2_layer_call_fn_29126oqrstuvA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ@Î
G__inference_sequential_3_layer_call_and_return_conditional_losses_26379wxyz{|I¢F
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
G__inference_sequential_3_layer_call_and_return_conditional_losses_26399wxyz{|I¢F
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
G__inference_sequential_3_layer_call_and_return_conditional_losses_29167zwxyz{|A¢>
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
G__inference_sequential_3_layer_call_and_return_conditional_losses_29206zwxyz{|A¢>
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
,__inference_sequential_3_layer_call_fn_26437uwxyz{|I¢F
?¢<
2/
conv2d_3_inputÿÿÿÿÿÿÿÿÿ@
p

 
ª " ÿÿÿÿÿÿÿÿÿ@@@¥
,__inference_sequential_3_layer_call_fn_26474uwxyz{|I¢F
?¢<
2/
conv2d_3_inputÿÿÿÿÿÿÿÿÿ@
p 

 
ª " ÿÿÿÿÿÿÿÿÿ@@@
,__inference_sequential_3_layer_call_fn_29223mwxyz{|A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ@
p

 
ª " ÿÿÿÿÿÿÿÿÿ@@@
,__inference_sequential_3_layer_call_fn_29240mwxyz{|A¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ@
p 

 
ª " ÿÿÿÿÿÿÿÿÿ@@@Ñ
G__inference_sequential_4_layer_call_and_return_conditional_losses_26708	}~H¢E
>¢;
1.
conv2d_4_inputÿÿÿÿÿÿÿÿÿ@@
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 Ñ
G__inference_sequential_4_layer_call_and_return_conditional_losses_26727	}~H¢E
>¢;
1.
conv2d_4_inputÿÿÿÿÿÿÿÿÿ@@
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 È
G__inference_sequential_4_layer_call_and_return_conditional_losses_29293}	}~@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 È
G__inference_sequential_4_layer_call_and_return_conditional_losses_29331}	}~@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ@@
 ¨
,__inference_sequential_4_layer_call_fn_26764x	}~H¢E
>¢;
1.
conv2d_4_inputÿÿÿÿÿÿÿÿÿ@@
p

 
ª "!ÿÿÿÿÿÿÿÿÿ@@¨
,__inference_sequential_4_layer_call_fn_26800x	}~H¢E
>¢;
1.
conv2d_4_inputÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ@@ 
,__inference_sequential_4_layer_call_fn_29348p	}~@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p

 
ª "!ÿÿÿÿÿÿÿÿÿ@@ 
,__inference_sequential_4_layer_call_fn_29365p	}~@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ@@Ô
G__inference_sequential_5_layer_call_and_return_conditional_losses_27059H¢E
>¢;
1.
conv2d_5_inputÿÿÿÿÿÿÿÿÿ@@
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 Ô
G__inference_sequential_5_layer_call_and_return_conditional_losses_27079H¢E
>¢;
1.
conv2d_5_inputÿÿÿÿÿÿÿÿÿ@@
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 Ì
G__inference_sequential_5_layer_call_and_return_conditional_losses_29406@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 Ì
G__inference_sequential_5_layer_call_and_return_conditional_losses_29445@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p 

 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ  
 «
,__inference_sequential_5_layer_call_fn_27117{H¢E
>¢;
1.
conv2d_5_inputÿÿÿÿÿÿÿÿÿ@@
p

 
ª "!ÿÿÿÿÿÿÿÿÿ  «
,__inference_sequential_5_layer_call_fn_27154{H¢E
>¢;
1.
conv2d_5_inputÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ  £
,__inference_sequential_5_layer_call_fn_29462s@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p

 
ª "!ÿÿÿÿÿÿÿÿÿ  £
,__inference_sequential_5_layer_call_fn_29479s@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ@@
p 

 
ª "!ÿÿÿÿÿÿÿÿÿ  Á
G__inference_sequential_6_layer_call_and_return_conditional_losses_27200vG¢D
=¢:
0-
flatten_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
G__inference_sequential_6_layer_call_and_return_conditional_losses_27210vG¢D
=¢:
0-
flatten_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
G__inference_sequential_6_layer_call_and_return_conditional_losses_29505o@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
G__inference_sequential_6_layer_call_and_return_conditional_losses_29518o@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_sequential_6_layer_call_fn_27230iG¢D
=¢:
0-
flatten_inputÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_6_layer_call_fn_27249iG¢D
=¢:
0-
flatten_inputÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_6_layer_call_fn_29527b@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ  
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_6_layer_call_fn_29536b@¢=
6¢3
)&
inputsÿÿÿÿÿÿÿÿÿ  
p 

 
ª "ÿÿÿÿÿÿÿÿÿÌ
E__inference_sequential_layer_call_and_return_conditional_losses_25348efghijG¢D
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
E__inference_sequential_layer_call_and_return_conditional_losses_25367efghijG¢D
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
E__inference_sequential_layer_call_and_return_conditional_losses_28815|efghijA¢>
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
E__inference_sequential_layer_call_and_return_conditional_losses_28853|efghijA¢>
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
*__inference_sequential_layer_call_fn_25404uefghijG¢D
=¢:
0-
conv2d_inputÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿ £
*__inference_sequential_layer_call_fn_25440uefghijG¢D
=¢:
0-
conv2d_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ 
*__inference_sequential_layer_call_fn_28870oefghijA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿ 
*__inference_sequential_layer_call_fn_28887oefghijA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ ß
#__inference_signature_wrapper_28143·1efghijklmnopqrstuvwxyz{|}~E¢B
¢ 
;ª8
6
input_1+(
input_1ÿÿÿÿÿÿÿÿÿ";ª8
6
sequential_6&#
sequential_6ÿÿÿÿÿÿÿÿÿ