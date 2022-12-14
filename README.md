# freMTPL2

Datenexploration befindet sich im `ExploreData.ipynb` Notebook, mit einem kleinen Beispiel Boosted Regression Tree.

Idealerweise würde zwei Regression Neural Networks benutzt, entweder zusammen trainiert oder getrennt.

* Das erste würde die poissische Claims/Jahr Verteilung von den unabhängigen Variabeln vorhersagen.
* Das zweite würde die durchschnittliche ClaimAmount von den unabhängigen Variabeln vorhersagen verhersagen.

Von den beiden NNs oder Decision Trees wäre es dann möglich pro Client pro Jahr eine erwartete Summe zu berechnen.

In `utils.py` gibt es ``Helper Functions'' für Datenanalyse, in `plot.py` werden verschiedene Funktionen für graphische Darstellung definiert und in `network.py` befinden sich Klassen und Funktionen für das Training von Neural Networks.

Die beide `run_XXX.py` Skripte würden die Modellen trainieren und evaluieren, und Run Optionen wären von Config Dateien gesteuert. 

Zeitdauer: 2 Stunden.
* Mehrheit der Zeit für explorativen Studien der Daten
* Prospektive Studien und eine grobe Struktur sind detailliert in den Dateien
