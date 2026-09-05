# Field guide: photographing dwarves for the query-domain experiment

Every reference photograph in this dataset is a Wikimedia Commons upload: good light,
considered framing, often a photographer who meant to document the statue. `RESULTS.md` names the
gap to a casual phone photo as the largest untested thing in the project. This is how we close it.

**What gets measured.** Your photographs become *queries* against the existing 1,691 reference
images. Nothing about the reference set changes. The number we want is how far top-1 accuracy falls
below the 93.1% the leave-one-out protocol reports — that difference is the domain gap, and right
now nobody knows if it is two points or thirty.

## How to shoot

- **Three to five photographs per statue**, and deliberately vary them. One straight-on at normal
  standing distance, one closer, one from an oblique angle, one wider with surroundings in frame.
  If the light is awkward, shoot it anyway — awkward light is the point.
- **Do not compose like a documentarian.** Hold the phone the way a tourist would. Do not clean the
  frame, wait for people to move, or crouch to get the perfect angle. A too-careful photo measures
  nothing we do not already know.
- **Leave GPS on** if you are willing. Every one of these statues is now geolocated, so photo
  coordinates let the pipeline propose which statue each photo belongs to instead of you sorting
  144 files by hand.
- Portrait or landscape, either is fine. Do not crop or filter afterwards.

## Where the files go

One directory per statue, named by its dataset ID. The directories already exist:

```
data/field-queries/
├── Q136343586/          <- Binio, for example
│   ├── IMG_4821.jpg
│   ├── IMG_4822.jpg
│   └── IMG_4823.jpg
└── C-slupniki-solne-dwarfs-wroclaw/
    └── ...
```

Filenames do not matter — keep whatever the phone assigns. The directory name is what links a
photo to its statue, and the IDs below are exactly the directory names. `data/field-queries/` is
git-ignored, like every other image directory.

## The route

Ordered as a walk starting from the Rynek, about **3.0 km** for the core set. Coordinates marked
`~` are derived from where photographers stood, so they are accurate to roughly 9 m — close enough
to find a statue, not a survey marker.

### Core — 19 statues, and a complete experiment on its own

1. **Sprinkler dwarf (Wrocław)** — control, 14 reference photos  
   `C-sprinkler-dwarf-wroclaw`  
   [51.11009, 17.03204 ~](https://www.google.com/maps/search/?api=1&query=51.110089,17.032040)

2. **Ślepak** — **3 errors**, 3 reference photos  
   `C-slepak-dwarf-wroclaw`  
   [51.10966, 17.03169 ~](https://www.google.com/maps/search/?api=1&query=51.109661,17.031686)

3. **Głuchak** — **4 errors**, 4 reference photos  
   `C-gluchak-dwarf-wroclaw`  
   [51.10966, 17.03169 ~](https://www.google.com/maps/search/?api=1&query=51.109661,17.031686)

4. **Syzyfki** — **1 errors**, 31 reference photos  
   `Q136001355`  
   [51.10889, 17.03278](https://www.google.com/maps/search/?api=1&query=51.108889,17.032778)

5. **Capgeminiusz Programista** — **3 errors**, 12 reference photos  
   `Q136341125`  
   [51.10867, 17.03298](https://www.google.com/maps/search/?api=1&query=51.108667,17.032980)

6. **Słupniki Świdnickie** — **7 errors**, 9 reference photos  
   `C-slupniki-swidnickie-dwarves-wroclaw`  
   [51.10843, 17.03276 ~](https://www.google.com/maps/search/?api=1&query=51.108429,17.032758)

7. **Grajek** — **5 errors**, 6 reference photos  
   `C-grajek-dwarf-wroclaw`  
   [51.10862, 17.03506 ~](https://www.google.com/maps/search/?api=1&query=51.108623,17.035058)

8. **Grajek i Meloman** — **8 errors**, 11 reference photos  
   `C-grajek-i-meloman-dwarves-wroclaw`  
   [51.10862, 17.03506 ~](https://www.google.com/maps/search/?api=1&query=51.108623,17.035058)

9. **Meloman** — **3 errors**, 9 reference photos  
   `C-meloman-dwarf-wroclaw`  
   [51.10862, 17.03506 ~](https://www.google.com/maps/search/?api=1&query=51.108623,17.035058)

10. **Słupniki Oławskie** — **7 errors**, 10 reference photos  
   `C-slupniki-olawskie-dwarves-wroclaw`  
   [51.10852, 17.03529 ~](https://www.google.com/maps/search/?api=1&query=51.108525,17.035291)

11. **Kowal** — **2 errors**, 12 reference photos  
   `Q65742089`  
   [51.11052, 17.03357](https://www.google.com/maps/search/?api=1&query=51.110522,17.033567)

12. **WrocLovek** — control, 14 reference photos  
   `C-wroclovek-dwarf-wroclaw`  
   [51.11114, 17.03074 ~](https://www.google.com/maps/search/?api=1&query=51.111136,17.030742)

13. **Śpioch** — **1 errors**, 23 reference photos  
   `C-spioch-dwarf-wroclaw`  
   [51.11116, 17.03068 ~](https://www.google.com/maps/search/?api=1&query=51.111161,17.030681)

14. **Więzień** — control, 17 reference photos  
   `C-wiezien-dwarf-wroclaw`  
   [51.11247, 17.03247 ~](https://www.google.com/maps/search/?api=1&query=51.112468,17.032470)

15. **Słupniki Solne** — **8 errors**, 11 reference photos  
   `C-slupniki-solne-dwarfs-wroclaw`  
   [51.10941, 17.02931 ~](https://www.google.com/maps/search/?api=1&query=51.109410,17.029311)

16. **Puszczający Stateczki** — **3 errors**, 3 reference photos  
   `Q136290068`  
   [51.10532, 17.03313](https://www.google.com/maps/search/?api=1&query=51.105320,17.033131)

17. **Zbierający Wodę** — **5 errors**, 3 reference photos  
   `Q136290218`  
   [51.10532, 17.03313](https://www.google.com/maps/search/?api=1&query=51.105320,17.033131)

18. **Karmiący Ptaki** — **2 errors**, 3 reference photos  
   `Q136276947`  
   [51.10532, 17.03313](https://www.google.com/maps/search/?api=1&query=51.105320,17.033131)

19. **Pracz Odrzański** — control, 14 reference photos  
   `C-pracz-odrzanski-dwarf-wroclaw`  
   [51.11356, 17.04022 ~](https://www.google.com/maps/search/?api=1&query=51.113561,17.040224)


### Why these specific statues

The core set is not a random sample. It is built to answer two questions at once:

- **Słupniki Solne, Słupniki Oławskie, Słupniki Świdnickie** — the Słupniki pillar dwarves — the top confusion for both backbones.
- **Zbierający Wodę, Puszczający Stateczki, Karmiący Ptaki** — the water-themed trio — the cluster that dominated at 23 classes.
- **Ślepak, Głuchak** — the sculpted pair the projection picks out unprompted.
- **Grajek i Meloman, Grajek, Meloman** — a group class competing with its own members.

Those eleven are where both backbones already make their mistakes on *clean* photographs. If a
phone photo degrades the hard cases disproportionately, that is a different and more interesting
finding than a uniform drop, and it is only visible if the hard cases are in the query set.

The remaining eight are controls: well-photographed statues in the same streets that neither
backbone has ever confused. Without them a drop could just mean "these statues are hard", and the
comparison would be confounded by hard statues also standing somewhere awkward.

### Extended — if you have a longer day

1. **Bartonik** — **2 errors**, 7 reference photos  
   `C-bartonik-dwarf-wroclaw`  
   [51.10991, 17.03214 ~](https://www.google.com/maps/search/?api=1&query=51.109914,17.032137)

2. **Pomagajek** — **2 errors**, 7 reference photos  
   `C-pomagajek-dwarf-wroclaw`  
   [51.10975, 17.03175 ~](https://www.google.com/maps/search/?api=1&query=51.109749,17.031746)

3. **Prezentuś** — **2 errors**, 6 reference photos  
   `C-prezentus-dwarf-wroclaw`  
   [51.10977, 17.03078 ~](https://www.google.com/maps/search/?api=1&query=51.109772,17.030781)

4. **Leszko** — control, 10 reference photos  
   `C-leszko-dwarf-wroclaw`  
   [51.10993, 17.03086 ~](https://www.google.com/maps/search/?api=1&query=51.109929,17.030864)

5. **Wiesiek Partnerski** — **1 errors**, 3 reference photos  
   `C-wiesiek-partnerski-dwarf-wroclaw`  
   [51.11018, 17.03122 ~](https://www.google.com/maps/search/?api=1&query=51.110176,17.031223)

6. **Życzliwek** — **2 errors**, 10 reference photos  
   `C-zyczliwek-dwarf-wroclaw`  
   [51.11058, 17.03129 ~](https://www.google.com/maps/search/?api=1&query=51.110578,17.031292)

7. **Suvenirek** — **3 errors**, 11 reference photos  
   `C-suvenirek-dwarf-wroclaw`  
   [51.11068, 17.03059 ~](https://www.google.com/maps/search/?api=1&query=51.110678,17.030592)

8. **Pocztowiec** — control, 9 reference photos  
   `C-pocztowiec-dwarf-wroclaw`  
   [51.11089, 17.03067 ~](https://www.google.com/maps/search/?api=1&query=51.110886,17.030675)

9. **Pożarki** — **2 errors**, 9 reference photos  
   `C-pozarki-dwarves-wroclaw`  
   [51.11120, 17.03045 ~](https://www.google.com/maps/search/?api=1&query=51.111200,17.030449)

10. **Chrapek** — **2 errors**, 7 reference photos  
   `C-chrapek-dwarf-wroclaw`  
   [51.11167, 17.02972 ~](https://www.google.com/maps/search/?api=1&query=51.111667,17.029722)

11. **Podróżnik dwarf (Zwolańska–Hołod), Wrocław** — control, 11 reference photos  
   `C-podroznik-dwarf-zwolanskaholod-wroclaw`  
   [51.11225, 17.03008 ~](https://www.google.com/maps/search/?api=1&query=51.112253,17.030075)

12. **Bankomatki** — **1 errors**, 8 reference photos  
   `C-bankomatki-dwarfs-wroclaw`  
   [51.11171, 17.03407 ~](https://www.google.com/maps/search/?api=1&query=51.111713,17.034072)

13. **Rogalik** — **1 errors**, 6 reference photos  
   `C-rogalik-dwarf-wroclaw`  
   [51.11073, 17.03356 ~](https://www.google.com/maps/search/?api=1&query=51.110732,17.033558)

14. **Ciastuś i Amorinek** — control, 8 reference photos  
   `C-ciastus-i-amorinek-dwarves-wroclaw`  
   [51.11042, 17.03363 ~](https://www.google.com/maps/search/?api=1&query=51.110417,17.033626)

15. **Klucznik** — control, 9 reference photos  
   `C-klucznik-dwarf-wroclaw`  
   [51.10969, 17.03434 ~](https://www.google.com/maps/search/?api=1&query=51.109688,17.034340)

16. **Motocyklista** — control, 9 reference photos  
   `C-motocyklista-dwarf-wroclaw`  
   [51.10957, 17.03453 ~](https://www.google.com/maps/search/?api=1&query=51.109573,17.034534)

17. **Pomnik Pomarańczowej Alternatywy** — control, 11 reference photos  
   `Q11823412`  
   [51.10720, 17.03200](https://www.google.com/maps/search/?api=1&query=51.107200,17.032000)

18. **Wodne** — **1 errors**, 8 reference photos  
   `C-wodne-dwarves-wroclaw`  
   [51.10661, 17.03354 ~](https://www.google.com/maps/search/?api=1&query=51.106613,17.033544)

19. **Aktor** — control, 8 reference photos  
   `Q136276711`  
   [51.10539, 17.03321](https://www.google.com/maps/search/?api=1&query=51.105393,17.033215)

20. **Wierzbownik** — **1 errors**, 3 reference photos  
   `Q136290183`  
   [51.10532, 17.03313](https://www.google.com/maps/search/?api=1&query=51.105320,17.033131)

21. **Tescoma** — **1 errors**, 6 reference photos  
   `C-tescoma-dwarf-wroclaw`  
   [51.10254, 17.03700 ~](https://www.google.com/maps/search/?api=1&query=51.102545,17.037003)

22. **Turysta** — **2 errors**, 9 reference photos  
   `C-turysta-dwarf-wroclaw`  
   [51.10950, 17.03064 ~](https://www.google.com/maps/search/?api=1&query=51.109497,17.030639)

23. **Powerek** — **1 errors**, 4 reference photos  
   `C-powerek-dwarf-wroclaw`  
   [51.10956, 17.03034 ~](https://www.google.com/maps/search/?api=1&query=51.109556,17.030335)

24. **Wypłatnik** — control, 9 reference photos  
   `C-wyplatnik-dwarf-wroclaw`  
   [51.10979, 17.03021 ~](https://www.google.com/maps/search/?api=1&query=51.109793,17.030206)

25. **Bankuś Pieniążek** — **2 errors**, 6 reference photos  
   `C-bankus-pieniazek-dwarf-wroclaw`  
   [51.10999, 17.03029 ~](https://www.google.com/maps/search/?api=1&query=51.109988,17.030289)

26. **Dialogomir** — **1 errors**, 6 reference photos  
   `C-dialogomir-dwarf-wroclaw`  
   [51.10967, 17.02920 ~](https://www.google.com/maps/search/?api=1&query=51.109667,17.029195)

27. **Tolerance** — control, 9 reference photos  
   `C-tolerance-dwarves-wroclaw`  
   [51.10833, 17.02603 ~](https://www.google.com/maps/search/?api=1&query=51.108331,17.026033)

28. **Szomol** — **2 errors**, 10 reference photos  
   `C-szomol-dwarf-wroclaw`  
   [51.10735, 17.02591 ~](https://www.google.com/maps/search/?api=1&query=51.107352,17.025909)

29. **Profesor Medyk** — control, 12 reference photos  
   `C-profesor-medyk-dwarf-wroclaw`  
   [51.11376, 17.03361 ~](https://www.google.com/maps/search/?api=1&query=51.113756,17.033613)

30. **Kapitańskie Bliźniaki** — control, 9 reference photos  
   `C-kapitanskie-blizniaki-dwarves-wroclaw`  
   [51.11428, 17.04208 ~](https://www.google.com/maps/search/?api=1&query=51.114277,17.042077)

31. **Drukarz Kacper** — **1 errors**, 17 reference photos  
   `C-drukarz-kacper-dwarf-wroclaw`  
   [51.11456, 17.04333 ~](https://www.google.com/maps/search/?api=1&query=51.114556,17.043334)

32. **Botanik** — **2 errors**, 8 reference photos  
   `C-botanik-dwarf-wroclaw`  
   [51.11635, 17.04728 ~](https://www.google.com/maps/search/?api=1&query=51.116347,17.047283)

33. **Golasek dwarf in Wrocław** — **1 errors**, 16 reference photos  
   `C-golasek-dwarf-in-wroclaw`  
   [51.11566, 17.05135 ~](https://www.google.com/maps/search/?api=1&query=51.115660,17.051354)

34. **Rozkwietnik** — **1 errors**, 5 reference photos  
   `C-rozkwietnik-dwarf-wroclaw`  
   [51.12248, 17.04038 ~](https://www.google.com/maps/search/?api=1&query=51.122482,17.040380)


## When you get back

Copy the directories across and tell me. I will build a query manifest, embed the photographs with
the same pinned backbones, and score them against the existing references — reporting the domain
gap overall and separately for the confusable clusters against the controls.
