# Violin Fingering Generation

We build an intelligent system which generates fingerings for violin music in an interactive way. Instead of fully-automatic generation of violin fingerings, the system provides multiple generation paths and yield adaptable fingering arrangements for users. A new violin dataset with fingering annotations is also proposed. For more details, please refer to ["Positioning Left-hand Movement in Violin Performance: A System and User Study of Fingering Pattern Generation" (IUI 2021)](https://dl.acm.org/doi/abs/10.1145/3397481.3450661?sid=SCITRUS).

**TNUA Violin Fingering Dataset**
The dataset contains 10 violin pieces and the corresponding note-by-note annotations by 10 professional musicians. The annotations specify the detailed performance attributes of each note, including pitch, metric onset, duration, beat type, string designation, hand position, and finger choice.

The filenames are formatted as [violinist_id]_[composer:piece_id:section_id].csv

For example, the filename *vio1_beeth2_1.csv* indicates that the fingerings are annotated by violinist 1 on Beethoven's Violin Sonata No. 6, mvt. 3, Theme.

The 10 violin pieces are:
1. Bach: Sonatas and Partitas for Solo Violin, Partita No. 2 in D minor, BWV 1004, Allemanda
2. Bach: Sonatas and Partitas for Solo Violin, Partita No. 3 in E major, BWV 1006, Preludio
3. Mozart: Violin Concerto No. 3 in G major, K. 216, mvt. 1, mm. 38-94
4. Mozart: Violin Concerto No. 3 in G major, K. 216, mvt. 3, mm. 40-234
5. Beethoven: Violin Sonata No. 5 in F major, Op. 24, mvt. 1, mm. 1-86
6. Beethoven: Violin Sonata No. 6 in A major, Op. 30-1, mvt. 3, Theme & Variation 2
7. Elgar: Salut d'Amour, Op. 12
8. Mendelssohn: Violin Concerto in E minor, Op. 64, mvt. 1, mm. 1-131
9. Yu-hsien Teng: Bāng Chhun-hong (Taiwanese folk music) / 鄧雨賢:望春風
10. Yu-hsien Teng: Ú-iā-hue (Taiwanese folk music) / 鄧雨賢:雨夜花

