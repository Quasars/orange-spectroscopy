Spectrum Calculator
=========================

Perform arithmetic operations between spectra. Between two datasets in tables *Primary Data* and *Secondary Data*, this widget
performs elementwise arithmetic operations.

**Inputs**

 - Primary Data: Input data set
 - Secondary Data: Input data set

**Outputs**

 - Data: Data table with the element-wise arithmetic operation performed.

Examples
---------

The widget takes two input tables, of the same shape and size. Note that the
operations performed by this widget are done directly on an element-by-element basis. If the order of elements has been modified somewhere upstream, then this widget will **not** account for this in how the operation is done, and your output may not be as expected.

In each of the following examples, we will use the following as the two input data tables:

<table>
<tr><th>Primary Data</th><th>Secondary Data</th></tr>
<tr><td>

|a|b|c|
|-|-|-|
|1|2|3|
|4|5|6|

</td><td>

|a|b|c|
|-|-|-|
|7|8|9|
|10|11|12|

</td></tr> </table>

### Addition

`Addition` performs a simple addition between elements:

<table>
<tr><th>Data</th></tr>
<tr><td>

|a|b|c|
|-|-|-|
|8|10|12|
|14|16|18|

</td></tr> </table>


### Subtraction

`Subtraction` subtracts the secondary data table from the primary one:

<table>
<tr><th>Data</th></tr>
<tr><td>

|a|b|c|
|-|-|-|
|-6|-6|-6|
|-6|-6|-6|

</td></tr> </table>

### Division

`Division` divides the primary data table by the second:

<table>
<tr><th>Data</th></tr>
<tr><td>

|a|b|c|
|-|-|-|
|0.142|0.250|0.333|
|0.400|0.455|0.500|

</td></tr> </table>


### Multiplication

`Multiplication` multiplies the two data tables together:

<table>
<tr><th>Data</th></tr>
<tr><td>

|a|b|c|
|-|-|-|
|7|16|27|
|40|55|72|

</td></tr> </table>

### Scaled Multiplication

`Scaled Multiplication` scales a simple multiplication by an additional factor. For example, for an input factor of 2:

<table>
<tr><th>Data</th></tr>
<tr><td>

|a|b|c|
|-|-|-|
|14|32|54|
|80|110|144|

</td></tr> </table>


