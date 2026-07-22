Gate
====



Gate connections between widgets so large datasets don't flow at the same time

**Inputs**

- Data: input dataset(s)

**Outputs**

- Data: input dataset(s). No data is processed or modified by this widget.


This widget may be useful for e.g. large branched workflows, where changing input data upstream could be computationally expensive.
Putting a gating widget at significant junctures in the pipeline stops the flow of data until the gate is opened.


