# Violin Fingering Generation

We build an intelligent system which generates fingerings for violin music in an interactive way. Instead of fully-automatic generation of violin fingerings, the system provides multiple generation paths and yield adaptable fingering arrangements for users. A new violin dataset with fingering annotations is also proposed. For more details, please refer to ["Positioning Left-hand Movement in Violin Performance: A System and User Study of Fingering Pattern Generation" (IUI 2021)](https://dl.acm.org/doi/abs/10.1145/3397481.3450661?sid=SCITRUS).

**TNUA Violin Fingering Dataset**
The dataset contains 10 violin pieces and the corresponding note-by-note annotations by 10 professional musicians. The annotations specify the detailed performance attributes of each note, including pitch, metric onset, duration, beat type, string designation, hand position, and finger choice.

The filenames are formatted as vio[violinist_id]_[composer:piece_id:section_id].csv

For example, the filename * vio1_beeth2_1.csv * indicates that the fingerings are annotated by violinist 1 on Beethoven’s piece 2, section 1.
