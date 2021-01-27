# Peak Labeling

The peak labeling menu enables allows for individual or automatic peak labeling.
This processes is achieved through the use of the Scyfi find peaks toolkit.
Using this, it is possible to find find peaks and use adjustable parameters 
to adjust what is labeled as a peak even when plotting multiple spectra with
overlapping peak positions. This process is limited by the clarity of data.
As such, while it can preform on noisy datasets, it will yield clear results
after data has been filtered.  


Parameters:

Prominence - Sets a minimum value for the required prominence of a peak to be labeled.

Minimum Peak Height - Sets a minimum Y value for peaks.

Maximum Peak Height - Sets a maximum Y value for peaks.

Minimum Distance Between Lines - Used to limit clusters of peaks. The value is 
input as the minimum allowable X distance between two labeled peaks. 
