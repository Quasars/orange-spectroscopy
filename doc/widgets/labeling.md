# Peak Labeling

The peak labeling menu allows for individual or automatic peak labeling of spectra.
Automatic labeling uses the Scipy find peaks module 
and is limited by the clarity within the data. 
As such, while it can preform on noisy datasets, it will yield clear results
after data has been filtered. Added peaks can be remvoed by double right clicking. 


Parameters:

Prominence - Sets a minimum value for the required prominence of a peak to be labeled.

Minimum Peak Height - Sets a minimum Y value for peaks.

Maximum Peak Height - Sets a maximum Y value for peaks.

Minimum Distance Between Lines - Used to limit clusters of peaks. The value is 
input as the minimum allowable X distance between two labeled peaks 
currently selects the first of neighboring peaks.
