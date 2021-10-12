# Violin Fingering Generation

We built an intelligent system to generate fingerings for violin music in an interactive way. Instead of fully-automatic generation of violin fingerings, the system provides multiple generation paths and yield adaptable fingering arrangements for users. A new violin dataset with fingering annotations is also proposed. For more details, please refer to ["Positioning Left-hand Movement in Violin Performance: A System and User Study of Fingering Pattern Generation" (IUI 2021)](https://dl.acm.org/doi/abs/10.1145/3397481.3450661?sid=SCITRUS).

## TNUA Violin Fingering Dataset ##
The dataset contains 10 violin pieces and the corresponding note-by-note annotations by 10 professional musicians. The annotations specify the detailed performance attributes of each note, including pitch, metrical onset, duration, beat type, string designation, hand position, and finger choice.

The filenames are formatted as *[violinist_id]_[piece_id].csv*. For example, the filename *vio1_bach1.csv* indicates that the fingerings are annotated by violinist 1 on Bach's Partita No. 2.

The 10 violin pieces are:
1. Bach: Sonatas and Partitas for Solo Violin, Partita No. 2 in D minor, BWV 1004, Allemanda (bach1)
2. Bach: Sonatas and Partitas for Solo Violin, Partita No. 3 in E major, BWV 1006, Preludio (bach2)
3. Mozart: Violin Concerto No. 3 in G major, K. 216, mvt. 1, Solo (mozart1)
4. Mozart: Violin Concerto No. 3 in G major, K. 216, mvt. 3 (mozart2_1 mm. 41-234, mozart2_2 mm. 252-393)
5. Beethoven: Violin Sonata No. 5 in F major, Op. 24, mvt. 1 (beeth1)
6. Beethoven: Violin Sonata No. 6 in A major, Op. 30-1, mvt. 3 (beeth2_1 Theme - Vaiation 3, beeth2_2 Varitaion 4 - Varitaion 6)
7. Elgar: Salut d'Amour, Op. 12 (elgar)
8. Mendelssohn: Violin Concerto in E minor, Op. 64, mvt. 1 (mend1 mm. 2-141, mend2 mm. 141-313, mend3 mm. 313-463)
9. Yu-hsien Teng: Bāng Chhun-hong (Taiwanese folk music) / 鄧雨賢: 望春風 (wind)
10. Yu-hsien Teng: Ú-iā-hue (Taiwanese folk music) / 鄧雨賢: 雨夜花 (flower)

## Violin Fingering Generation Model
Bi-directional Long Short-Term Memory Network (BLSTM) with a softmax layer is used to predict the probabilities of string and position for playing each violin note, and three modules are employed to make the output desisions of string designation, hand position, as well as finger choice. For intput of the model, each note is represented by its pitch, onset, duration, and beat type.

## Requirements
 * python >= 3.6.4
 * tensorflow >= 1.8.0
 * numpy >= 1.16.2
 * sklearn >= 0.19.1
 * pretty_midi >= 0.2.9

## Graphical User Interface
We have developed a simple GUI that allows users to interact with the violin fingering generation model. The GUI is designed to recommend violin fingerings on sheet music (in the MusicXML format) with three pre-defined generation modes and export the annotated scores as PDF or MusicXML files. To realize a customized recommendation, the GUI can also take user-specified fingerings as optional inputs. 

The GUI is proposed along with the paper: ["An Interactive Automatic Violin Fingering Recommendation Interface" (IEEE AIVR 2021)](...).