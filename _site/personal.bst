%%  DINALPHA.BST         ---  á-Version v. 9/11/94 K.F.Lorenzen
%%
%%  entwickelt aus BibTeX standard bibliography style `alpha'
%%  gemaess der deutschen Zitiernorm DIN 1505 Teil 2
%%        von Klaus F. Lorenzen, Hamburg  5/2/94 bis 28/3/94 
%%        Jobst-Hartmut Lueddecke, Hamburg  habe ich fuer Verbesserungsvor-
%%        schlaege und stete Unterstuetzung bei der Implementierung
%%        auf der DEC-Rechenanlage des RZ der FH Hamburg zu danken
%%        Oren Patashnik, dem BIBTEX-Schoepfer zugeeignet
%%  Dieser style produziert Literaturzitate in Literaturverzeichnissen
%%  gemaess der deutschen Norm DIN 1505, Teil 2 vom Jan. 1984.
%%  Es gibt eine Reihe prinzipieller Unterschiede zwischen der US- und 
%%  der deutschen Zitierkonvention. Daher ist auch keine 1-1-deutige  
%%  Abbildung der beiden Regelwerke aufeinander moeglich. 
%%  Dennoch laesst sich in fast allen Faellen problemlos ein Original-bib-file
%%  (d.i. die Datenbank, die die bibliographischen Informationen enthaelt)
%%  wahlweise nach US-Norm oder deutscher DIN-Norm verarbeiten. 
%%  [Beispiel: Produzieren Sie mit der XAMPL.bib-Datenbank aus dem Original-
%%  paket 2 verschiedene Literaturverzeichnisse.]     Zu Gunsten
%%  der Allgemeingueltigkeit von bib-files ist bei den Publikationstypen
%%  (entry-types) und den bibliographischen Kategorien (fields) in Zweifels-
%%  faellen immer (ja, mit endlichen Ausnahmen) die originale US-Bedeutung 
%%  beibehalten worden. Bei der Erfassung von Literaturquellen in bib-files 
%%  gelten folglich die in der TEX-Literatur veroeffentlichten Regeln.
%%  Kommt es dennoch zu kleineren "Schoenheitsfehlern" im fertig gesetzten 
%%  output, so koennen diese so gut wie immer durch eine leicht veraenderte 
%%  Erfassung im bib-file beseitigt werden.
%%
%%  Die entry-Funktionen booklet und phdthesis entfallen zugunsten der
%%  Funktionen book und mastersthesis.
%%
%%  Diese beta-Version ist an der FH Hamburg seit einem halben Jahr erfolgreich
%%  im Test und wird nun einem groesseren Anwenderkreis uebergeben. Eine aus-
%%  fuehrliche Dokumentation mit Beispielen ist in Vorbereitung. 
%%  ACHTUNG!  Aus Testgruenden, Zeitmangel und Faulheit sind noch nicht
%%  alle ueberfluessigen Fehlermeldungen des Original-style ausgemerzt.
%%  Ferner sind noch laengst nicht alle Programmverfeinerungen ausgekostet.
%%  Das fuehrt mitunter zu Wasserfallartigen Fehlermeldungen: vergessen Sie
%%  es einfach und schauen Sie sich lieber das fertige Literaturverzeichnis
%%  an!
%%  Aus dinalpha.bst sind analog zur Originalversion die anderen styles
%%  dinplain.bst, dinunsrt.bst, dinabbr.bst abgeleitet und einsatzbereit.
%%
%%  Kritik, Vorschlaege usw. bitte an 
%%          FH Hamburg,  Klaus F. Lorenzen, Grindelhof 30, 20146 Hamburg
%%          e-mail: lorenzen@rzbt.fh-hamburg.de
%%
% version 0.99a for BibTeX versions 0.99a or later, LaTeX version 2.09.
% Copyright (C) 1985, all rights reserved.
% Copying of this file is authorized only if either
% (1) you make absolutely no changes to your copy, including name, or
% (2) if you do make changes, you name it something other than
% dinbst.doc, dinplain.bst, dinunsrt.bst, dinalpha.bst, and dinabbrv.bst.
% This restriction helps ensure that all standard styles are identical.
%% ==>  The file btxbst.doc has the original documentation for style 'alpha'.
%%      The file dinbst.doc has the documentation for this style (`dinalpha').
%%      dinbst.doc is not yet delivered (9/11/94), Lo
%%


%% Weitere Aenderungen: Raphael Bielmeier (bielmeir@hrz.uni-kassel.de)
%% ===================================================================
%% - Dissertation (PhD Thesis) wieder aufgenommen
%% - Buchstabenlabels in Nummern geaendert

%% Anmerkung (Raphael Bielmeier, 05.03.96): Wie weiter unten erwaehnt,
%% bei inproceedings (Konferenzberichte), welche mehrbaendig sind,
%% aufpassen: BOOKTITLE durch SERIES ersetzen!! 

ENTRY
  { address      %% 
    author       %% 
    booktitle    %%
    chapter      %%
    edition      %% 
    editor       %% 
    howpublished %%
    institution  %%
    isbn         %% neues Feld: Standardnr fuer Buecher
    issn         %%  - " -    :      -"-        Zeitschriften u. Serien
    journal      %% 
    key          %%
    month        %% 
    note         %% 
    number       %% 
    organization %%
    pages        %% 
    publisher    %% 
    school       %% 
    series       %% 
    title        %% 
    type         %% 
    volume       %% 
    year         %% 
  }
  {}
  { label extra.label sort.label }

INTEGERS { output.state before.all mid.sentence after.sentence after.block }

%%$$$ die folg. BOOLE'sche VAR steuern d. Ausg. ": " nach Urheber-Feldern
%%    und ". - " vor ISBN oder Anmerkungen (NOTE)

INTEGERS { colon.after  period.dash }

FUNCTION {init.state.consts}
{ #0 'before.all :=
  #1 'mid.sentence :=
  #2 'after.sentence :=
  #3 'after.block :=
  #4 'colon.after :=
  #5 'period.dash :=
}
INTEGERS { zahl  lang }  

STRINGS { h s t u v }    

%%$$$ neue DIN-Funktion, 16/2/94

FUNCTION {output.nonnull}
{ 's :=
  output.state after.block =
      { add.period$  write$
	newline$
	"\newblock " write$
      }
      { output.state before.all =
	  {  write$ }     
	  {  output.state colon.after =
	       { ": " * write$
		 newline$
		 "\newblock " write$
	       }
	       { output.state period.dash =
		   { ". -- " *  write$
		     newline$
		     "\newblock " write$
		   }
		   { output.state mid.sentence =
		       { ", " * write$ }
		       { " "  * write$ 
			 newline$
			 "\newblock " write$
		       }
		     if$
		   }
		if$
	       }
	    if$
	   }     
	 if$
	 after.block 'output.state :=
       }
     if$
     s 
}

FUNCTION {output}
{ duplicate$ empty$
    'pop$
    'output.nonnull
  if$
}

FUNCTION {output.check}
{ 't :=
  duplicate$ empty$
    { pop$ "empty " t * " in " * cite$ * warning$ }
    'output.nonnull
  if$
}

FUNCTION {output.bibitem}
{ newline$
  "\bibitem[" write$
  label write$
  "]{" write$
  cite$ write$
  "}" write$
  newline$
  ""
  before.all 'output.state :=
}

FUNCTION {fin.entry}   %%$$$  nach DIN neu 16/2/94
{ write$
  newline$
}

FUNCTION {set.period.dash}    %% Wenn ein ". - " die Satzteile trennen soll.!
{ output.state before.all =
    'skip$
    { period.dash  'output.state := }
  if$
}

%%$$$ neu 16/2/94
%%   prueft, ob ISBN- oder NOTE-Feld vh. ist und setzt dann ". - " davor.
%%   wenn aber ein PAGES-Feld ausgegeben wurde, dann kommt nichts davor.

FUNCTION {set.period.dash.check}  
{ empty$                            
    'skip$
    'set.period.dash  
  if$
}

FUNCTION {set.colon.after}   %%$$$ Wenn ein ": " d. Satzteile trennen soll!
{ output.state before.all =
    'skip$
    { colon.after 'output.state  := }
  if$
}

%% neu / alt 17/2/94          Wenn ein " " die Satzteile trennen soll.!
FUNCTION {new.sentence}
{ output.state before.all =
     'skip$
     { after.sentence 'output.state := }
  if$
}

%% neu 17/2/94                 Wenn ein ", " die Satzteile trennen soll.!
FUNCTION { part.of.sentence }
{ output.state before.all =
     'skip$
     { mid.sentence 'output.state := }
  if$
}


FUNCTION {not}
{   { #0 }
    { #1 }
  if$
}

FUNCTION {and}
{   'skip$
    { pop$ #0 }
  if$
}

FUNCTION {or}
{   { pop$ #1 }
    'skip$
  if$
}

FUNCTION {new.sentence.checka}
{ empty$
    'skip$
    'new.sentence
  if$
}

FUNCTION {field.or.null}
{ duplicate$ empty$
    { pop$ "" }
    'skip$
  if$
}

FUNCTION {emphasize}
{ duplicate$ empty$
    { pop$ "" }
    { "{\em " swap$ * "\/}" * }
  if$
}

INTEGERS { nameptr namesleft numnames }

STRINGS { fkt }

%% neu, setzt Autor/Hrsg. in Kapitaelchen 9/3/94
%%FUNCTION { capitalize }
%% { duplicate$ empty$
%%    { pop$ "" }
%%    { "{\sc " swap$ * "}" * }
%%  if$
%%}

%% neu, setzt Autor/Hrsg. in Kapitaelchen 9/3/94
FUNCTION { capitalize }
 { duplicate$ empty$
    { pop$ "" }
    { "{" swap$ * "}" * }
  if$
}

%%$$$ DIN-Quellenangabe : spezieller unselbst. Teil ist erschienen "In: " 
%%                        dem bibliogr. selbst. Werk, z.B. Zeitschrift, Buch

%% 9/3/94
FUNCTION {inquelle}
{ duplicate$ empty$
    { pop$ "" }
%%    { emphasize "{\bf In: }" swap$ * }
    { emphasize "{ }" swap$ * }
  if$
}

%% Vorschlag von H.Lueddecke, um Adelspraedikate beim Sortieren
%% nach den(m) Vornamen aufzufuehren. Lo, 2/11/94
FUNCTION {format.names}
{ 's :=
  "" 'u  :=
  #1 'nameptr :=
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    {
%vorher, Lue      s nameptr "{vv~}{ll}" format.name$ 't :=
       s nameptr "{ll}" format.name$ 't :=
       t capitalize 't :=
%vorher,Lue      s nameptr "{, ff}" format.name$ 'u := 
       s nameptr "{, ff}{ vv}" format.name$ 'u := 
       u  text.length$  'lang :=     
       #1 'zahl :=
       "" 'v :=
	 {  zahl  lang  <  }
	 {  u  zahl  #1  substring$  "~"  =  
	       {  v  ""  =
		     {  u  #1  zahl  #1  -  substring$  'v :=  }
		     'skip$
		  if$
		  v  u  zahl  #2  substring$  *  "."  *  'v :=  }
	       'skip$
	    if$
	    zahl  #1  +  'zahl :=  }
      while$
      v  ""  =
	 {  u  'v :=  }
	 'skip$
      if$
%%    der string fkt enthaelt " (Hrsg.)", wenn Editorfeld nicht leer ist
      t  v  *  fkt  *  't :=  %% Komma nach Nachnamen wird oben erledigt!
			      %% t enthaelt nun d. formatierten Nnamen, Vnamen
      nameptr #1 >
	 { namesleft #1 >
	     { " ; " * t * }
	     { numnames #2 >
		  { " " * }
		  'skip$
	       if$
	       t "{\sc others}" =
%%                  { "et~al." * }     %% Geschmackssache
		    { " [~u.~a.~] " * }%%   13/2/94
		    { " ; " * t * }
	       if$
	       }
	    if$  %% Ende der namesleft-Pruefung
	  }
	    't    
	   %% hierdurch wird bei jed. Schleifendurchgang das sich komplet-
	   %% tierende Zwischen-Namensergebnis wieder auf den stack gelegt 

       if$  %%  Ende der nameptr-Pruefung

      nameptr #1 + 'nameptr :=
      namesleft #1 -  'namesleft :=
    }
  while$  %%  Ende von { namesleft #0 > } ganz oben
  "" 'fkt :=  %% fkt wird zurueckgesetzt
}

%%$$$ geaendert 14/2/94
 
FUNCTION {format.authors}
{ author empty$
    { "" }
    { author format.names }
  if$
}

%%$$$ geaend. 20/2/94 Anpassung an DIN mit Autor + Hrsg. !!

FUNCTION {format.editors}
{ editor empty$
   { author empty$
       { "Weder Verfasser noch Hrsg. in " cite$ * warning$ }
       'skip$ 
     if$
    }
    { author empty$
	{ " (Hrsg.)" 'fkt :=  
	  editor format.names 
	}
	{ " (Hrsg.)" 'fkt :=  
	  " ; " * editor format.names *
	}
      if$
     }
   if$
 } 

%%$$$  Sonderfall: Herausgeber bei Typ incollection, 21/2/94
FUNCTION {format.ed.incoll}
{ editor empty$
     { "" }
     { " (Hrsg.)" 'fkt :=  
       editor format.names 
     }
  if$
} 

%%$$$ neu 22/2/94  Neben den Herausgebern wird auch die veranstaltende
%%       Koerperschaft, z.B. ein Institut, e. Gesellschaft, Organisat. genannt.

FUNCTION {format.proceed.organization}
{  organization empty$
      { "" }
      { " / " organization * " (Veranst.)" * }
   if$
}

FUNCTION {format.title}
{ title empty$
    { "" }
    { title }  %% Text so wie er dasteht im Feld title
  if$
}

FUNCTION {n.dashify}
{ 't :=
  ""
    { t empty$ not }
    { t #1 #1 substring$ "-" =
	{ t #1 #2 substring$ "--" = not
	    { "--" *
	      t #2 global.max$ substring$ 't :=
	    }
	    {   { t #1 #1 substring$ "-" = }
		{ "-" *
		  t #2 global.max$ substring$ 't :=
		}
	      while$
	    }
	  if$
	}
	{ t #1 #1 substring$ *
	  t #2 global.max$ substring$ 't :=
	}
      if$
    }
  while$
}

%% geaendert 24/2/94
FUNCTION {format.date}
{ year empty$
    { month empty$
	{ "" }
	{ "there's a month but no year in " cite$ * warning$
	  month
	}
      if$
    }
    { month empty$  %% b. Buechern nur Jahr, ohne Monat ausgeb. im Impressum
	  'year
	  { month " " * year * }
      if$
    }
  if$
}

%%
%%$$$  neue Fkt., 16/2/94 u. 14/3/94       das sog. Impressum
FUNCTION {format.address.publisher.year}
{  publisher empty$
      { address empty$
	  { year empty$  
	      { "" }
	      { year } 
	    if$
	  }
	  { "Es gibt einen Verlagsort, aber keinen Verlag in " cite$ * warning$
	      address ", " * format.date * 
	  }
	if$
      }
      { address empty$
	  { year empty$
	     { "Es gibt nur eine Verlagsangabe in " cite$ * warning$ 
		publisher
	     }
	     { publisher ", " * format.date * }
	    if$
	  }
	  { year empty$
	      { address " : " * publisher * }
	      { address " : " * publisher * ", " * format.date * }
	    if$
	  }
	if$
      }
  if$
}

FUNCTION {format.btitle}
{ title emphasize
}

FUNCTION {tie.or.space.connect}
{ duplicate$ text.length$ #3 <
    { "~" }
    { " " }
  if$
  swap$ * *
}

FUNCTION {either.or.check}
{ empty$
    'pop$
    { "can't use both " swap$ * " fields in " * cite$ * warning$ }
  if$
}

%% neu 8/3/94 in dieser Funkt. steckt im volume empty-Teil noch ein bug, der
%%            aber ignoriert werden kann; das Ergebnis ist ok.
FUNCTION {format.btitle.vol}
{ number empty$
   { series empty$
     { volume empty$
	{ title emphasize }
	{ title emphasize ". Bd." * volume tie.or.space.connect }
       if$
     }
     { volume empty$
	{ title emphasize }
	{ series emphasize ". Bd." * volume tie.or.space.connect  " : " * title * } 
       if$
     }
     if$%% series-test
   }
   { title emphasize }%% wenn number ex., dann immer title-Feld
  if$%% Ende number-test
}

%%$$$ neu 16/2/94
%%   Serien- / Reihentitel werden im Feld series gespeichert. Weist die
%%   Serie eine Zaehlung der Einzeltitel auf, gibt man entweder nach DIN alles 
%%   in das Feld series so ein: ---> TITEL DER SERIE  NR. (der Vorlage) <---
%%   z. B.  SERIES= { Mensch und Computer 12 }. 
%%   [ Die Nummer der Vorlage darf auch durch " ; " abgesetzt werden. ]
%%   oder:
%%   man gibt die Zaehlung in das Feld NUMBER ein, z.B. NUMBER = {12}.
%%   Achtung!!
%%   Bei mehrbaendigen Werken steht d. Gesamttitel im Feld SERIES und die
%%   Bandzaehlung im Feld VOLUME; NUMBER darf dann nicht besetzt sein!
%%   Anderenfalls liegt ein Erfassungsfehler vor, da sich Reihe u. mehrbd.
%%   Werk gegenseitig ausschliessen.

FUNCTION {format.series.number.din}
{ volume empty$
    { number empty$
	  { series empty$
	       { "" }%%   Ausstieg mit Nullstring
	       { "(" series * ")" * } %% d. Seriennr koennte auch gleich hier
				      %% im SERIES-Feld miterfasst werden
	    if$
	  }      
	  { series empty$
	       { "(" number tie.or.space.connect ")" * 
		 "there's a number but no series in " cite$ * warning$ 
	       }
	       { "(" series * number tie.or.space.connect ")" * }
	    if$
	  }
       if$
     }
     { series empty$
	  { "" }
	  { type$ "proceedings" =  %%   Sonderfall !
		{ "(" series * ")" * }
		{ "" }%%   Ausstieg mit Nullstring, s. Kommentar
	    if$
	  }%% bei gezaehlten Reihen MUSS die Reihennr. im Feld NUMBER stehen!
       if$ %% wenn also d. Feld VOLUME nicht leer ist, dann liegt falsche
      }     %% Erfassung vor und es erfolgt d. Ausstieg mit d. Nullstring!
  if$
}


%%$$$ 16/2/94
%%  Auflagenvermerke gibt man komplett, einschliesslich Abkuerzungen in 
%%  das Feld edition ein: ---> EDITION= { 3., erw. und verb. Aufl. }
%%  oder fremdsprachlich:      EDITION= { 2nd edition }

FUNCTION {format.edition}
{ edition empty$
    { "" }
    { edition } 
  if$
}

%%$$$ neu, 18/3/94
FUNCTION { format.isbn.issn }
{ isbn empty$
     { issn empty$
	  { "" }
	  { "ISSN" issn n.dashify tie.or.space.connect }
       if$
     }
     { "ISBN" isbn n.dashify tie.or.space.connect }
  if$
}

%%$$$ geaendert, 21/2/94  gibt Seitenzahl bei BOOK-Typ und verwandten T. aus
FUNCTION {format.pages.book}
{ pages empty$
    { "" }
%    { pages n.dashify tie.or.space.connect " S" *}
    { pages "Seiten" n.dashify  tie.or.space.connect }
  if$
}

%%$$$  alle anderen Seitenang. zB. Zeitschrft., INBOOK usw. a la Orig., 9/3/94
FUNCTION {format.pages}
{ pages empty$
    { "" }
    { "S." pages n.dashify  tie.or.space.connect } 
  if$
}

%%$$$ Angaben v. Jahrgang, Jahr, Heftnr., Seiten bei Artikel-Typ
%%    14/3/94

FUNCTION {format.vol.year.num.pages}
{ volume field.or.null
    year empty$
      { "Es gibt einen Jahrgang, aber kein Jahr in " cite$ * warning$ }
      { " (" year * ")" * * }
    if$
    month empty$
    'skip$
    { ", " month * * }
    if$
    number empty$
    'skip$
    { ", Nr. " number * *
%      volume empty$
%       { "there's a number but no volume in " cite$ * warning$ }
%       'skip$
%      if$
    }
    if$
  pages empty$
    'skip$
    { duplicate$ empty$
	{ pop$ format.pages }%% da pages leer, wird nur "" auf stack gelegt
	{ ", " format.pages * *}
      if$
    }
  if$
}

%% geaendert 21/2/94
FUNCTION {format.chapter.pages}
{ chapter empty$
    'format.pages
    { type empty$
	{ "Kapitel " }
	{ type }
      if$
      chapter tie.or.space.connect
      pages empty$
	'skip$
	{ ", " * format.pages * }
      if$
    }
  if$
}

%%$$$  geaendert 21/2/94
FUNCTION {format.in.ed.booktitle.din}
{ booktitle empty$
    { "" }
    { editor empty$
	{ volume empty$
	    { "{\bf In: }" booktitle emphasize * }
	    { "{\bf In: }" booktitle emphasize * 
	      " Bd." volume tie.or.space.connect * 
	    }
	  if$
	}
	{ volume empty$
	    { "{\bf In: }" format.ed.incoll * ": " * booktitle emphasize * }
	    { "{\bf In: }" format.ed.incoll * ": " * booktitle emphasize * 
	      " Bd." volume tie.or.space.connect * 
	    }
	  if$
	}
      if$
    }
  if$
}

%% geaendert 1/3/94
FUNCTION {format.thesis.type}
{ type empty$
    'skip$
    { pop$
      type 
    }
  if$
}

%% geaendert 23/2/94    i.Orig. wird zuerst die number, dann der type getestet
FUNCTION {format.tr.number.din}
{ type empty$
     { number empty$
	 { " -- Forschungsbericht" }   %%  bei Minimalangaben besser ohne "."!
	 { "(" number tie.or.space.connect "). -- Forschungsbericht" * }
       if$
     }
     { number empty$
	 { " -- " type * }             %%  bei Minimalangaben besser ohne "."!
	 { "(" number tie.or.space.connect "). -- " * type * }
       if$
     }
  if$
}


FUNCTION {format.article.crossref}
{ key empty$
    { journal empty$
	{ "need key or journal for " cite$ * " to crossref " * crossref *
	  warning$
	  ""
	}
	{ "{\bf In: }{\em " journal * "\/}" * }
      if$
    }
    { "{\bf In: }" key * }
  if$
  " (siehe \cite{" * crossref * "}" * ")" *
}

%%geaendert 7/3/94 und noch einmal durch Lueddecke, s.o.
FUNCTION {format.crossref.editor}
%vorher,Lue  { editor #1 "{vv~}{ll}" format.name$ " (Hrsg.)" *
 { editor #1 "{ll}" format.name$ " (Hrsg.)" *
   editor num.names$ duplicate$
   #2 >
    { pop$ " et~al." * }
    { #2 <
	'skip$
	{ editor #2 "{ff }{vv }{ll}{ jj}" format.name$ "others" =
	    { " et~al." * }
	    { " ; " * editor #2 "{vv~}{ll}" format.name$ * " (Hrsg.)" * }
	  if$
	}
      if$
    }
  if$
}


FUNCTION {format.book.crossref}
{ volume empty$
    { "empty volume in " cite$ * "'s crossref of " * crossref * warning$
      "(siehe "
    }
    { ". -- Bd." volume tie.or.space.connect
      " von " *
    }
  if$
  editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
	{ series empty$
	    { "need editor, key, or series for " cite$ * " to crossref " *
	      crossref * warning$
	      "" *
	    }
	    { "{\em " * series * "\/} (siehe " * }
	  if$
	}
	{ key * }
      if$
    }
    { format.crossref.editor * }
  if$
  " \cite{" * crossref * "}" * ")" *
}

FUNCTION {format.incoll.inproc.crossref}
{ editor empty$
  editor field.or.null author field.or.null =
  or
    { key empty$
	{ booktitle empty$
	    { "need editor, key, or booktitle for " cite$ * " to crossref " *
	      crossref * warning$
	      ""
	    }
	    { "{\bf In: }{\em " booktitle * "\/}" * }
	  if$
	}
	{ "{\bf In: }" key * }
      if$
    }
    { "{\bf In: }" format.crossref.editor * }
  if$
  " (siehe \cite{" * crossref * "}" * ")" *
}

%%geaendert
FUNCTION {article}
{ output.bibitem
  format.authors "author" output.check
  set.colon.after
  format.title "title" output.check
  crossref missing$
%%    { journal inquelle output.nonnull 
    { journal inquelle output.nonnull 
      new.sentence 
      format.vol.year.num.pages output
    }
    { format.article.crossref output.nonnull }
  if$
  note set.period.dash.check
  note output
  issn set.period.dash.check
  format.isbn.issn output
  fin.entry
}

%%$$$   geaendert, 20/2/94
FUNCTION {book}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors  format.editors output.nonnull
    }
  if$
  set.colon.after
  crossref missing$
    { format.btitle.vol "title" output.check }
    { format.btitle "title" output.check }
  if$
  format.edition output  
  format.address.publisher.year "publisher" output.check
  new.sentence
  crossref missing$
    { format.series.number.din  output }
    { format.book.crossref output.nonnull }
  if$
  new.sentence
  format.pages.book output
  note set.period.dash.check
  note output 
  isbn set.period.dash.check
  format.isbn.issn output
  fin.entry
}

%% geaendert 23/2/94
FUNCTION {inbook}
{ output.bibitem
%% unselbst. Teile eines Buches werden am Anf. genannt, dann d selbst. Quelle 
  chapter empty$
    { "Keine Kapitelangaben in " cite$ * warning$ }
    { type empty$
	 { "Kap. " }
	 { type }
      if$
     }
  if$
  chapter tie.or.space.connect " {\bf In: }" *
%%   -------- jetzt kommt der bibliogr. selbst. Teil
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors output.nonnull
    }
  if$
  set.colon.after
  format.btitle.vol "title" output.check
  crossref missing$
    { format.edition output
      format.address.publisher.year "publisher" output.check
      new.sentence
      format.series.number.din  output 
      note set.period.dash.check
      note output 
      isbn set.period.dash.check
      format.isbn.issn output
    }
    { format.book.crossref output.nonnull
      note set.period.dash.check
      note output 
    }
  if$
  part.of.sentence
  format.pages "pages" output.check
  fin.entry
}

%%  geaendert 21/2/94
FUNCTION {incollection}
{ output.bibitem
  format.authors "author" output.check
  set.colon.after
  format.title "title" output.check
  crossref missing$
    { format.in.ed.booktitle.din "booktitle" output.check
      format.edition output
      format.address.publisher.year "publisher" output.check
      new.sentence
      format.series.number.din  output %%$$$ 7/3/94
    }
    { format.incoll.inproc.crossref output.nonnull }
  if$
  note set.period.dash.check
  note output 
  isbn set.period.dash.check
  issn set.period.dash.check
  format.isbn.issn output
  part.of.sentence
  format.chapter.pages output
  fin.entry
}

%%  geaendert 22/2/94
FUNCTION {inproceedings}
{ output.bibitem
  format.authors "author" output.check
  set.colon.after
  format.title "title" output.check
  crossref missing$
    { format.in.ed.booktitle.din "booktitle" output.check
      volume empty$
	 'skip$
	 { " Bd." volume tie.or.space.connect output }
      if$
      address empty$
	{ organization new.sentence.checka
	  organization output
	  part.of.sentence
	  format.address.publisher.year output
	}
	{ format.address.publisher.year "publisher" output.check }
      if$
    }
    { format.incoll.inproc.crossref output.nonnull }
  if$
  note set.period.dash.check
  note output 
  isbn set.period.dash.check
  format.isbn.issn output
  part.of.sentence
  format.pages output
  fin.entry
}

%% FUNCTION {conference} { inproceedings }

%% geaendert, 21/2/94
FUNCTION {manual}
{ output.bibitem
  author empty$
    { organization empty$
	'skip$
	{ organization output.nonnull
	  address output
	}
      if$
    }
    { format.authors output.nonnull }
  if$
  set.colon.after
  format.btitle "title" output.check
  format.edition output
  author empty$
    { organization empty$
	{ address output
	  part.of.sentence
	}
	'skip$
      if$
    }
    { address ": " *  organization * output
      part.of.sentence
    }
  if$
  format.date output
  pages new.sentence.checka
  format.pages.book output
  note set.period.dash.check
  note output
  fin.entry
}


%%$$$%%   geaendert KFL, 17/2/94 Standard = "Diplomarbeit", sonst TYPE="anderer Typ"
FUNCTION {mastersthesis}
{ output.bibitem
  format.authors "author" output.check
  set.colon.after
  format.btitle "title" output.check
  address output
  part.of.sentence 
  school "school" output.check
  part.of.sentence
  "Diplomarbeit" format.thesis.type output.nonnull
  part.of.sentence
  format.date "year" output.check
  pages new.sentence.checka
  format.pages output
  note set.period.dash.check
  note output
  fin.entry
}

%%$$$%%   geaendert 02.03.96, Raphael Bielmeier:
%%$$$%%   "Dissertation" muá es geben, oder?! Kopiert von masterthesis
%%$$$%%   und gaendert.
FUNCTION {phdthesis}
{ output.bibitem
  format.authors "author" output.check
  set.colon.after
  format.btitle "title" output.check
  address output
  part.of.sentence 
  school "school" output.check
  part.of.sentence
  "Dissertation" format.thesis.type output.nonnull
  part.of.sentence
  format.date "year" output.check
  pages new.sentence.checka
  format.pages output
  note set.period.dash.check
  note output
  fin.entry
}

%% 26/2/94 hiermit werden u.a. Normen erfasst
FUNCTION {misc}
{ output.bibitem
  note empty$
     { title empty$ 
	  { "" }
	  { format.authors output
	    format.title output
	    howpublished output
	    format.date output 
	  } 
	if$
      }
      { note duplicate$ #1 #4 substring$ "Norm" =
	   { output new.sentence 
	     format.date output 
	     format.title output
	   }
	   { pop$ ""
	     author empty$
	       { format.editors output }
	       { format.authors  format.editors output.nonnull }
	     if$
	     set.colon.after
	     format.title output
	     howpublished output
	     format.date output 
	     note set.period.dash.check
	     note output
	   }
	 if$
       }
   if$
  fin.entry
}

%% geaendert 21/2/94
FUNCTION {proceedings}
{ output.bibitem
  editor empty$
    { organization empty$
	 { "" } 
	 { organization " (Veranst.)" * output }
      if$
    }
    { format.editors output.nonnull }
  if$
  set.colon.after
  format.btitle "title" output.check
  new.sentence
  editor empty$
    'skip$  
    {  format.proceed.organization output }
  if$
  new.sentence
  format.series.number.din emphasize output.nonnull
  volume empty$
    { "" output.nonnull } 
    { "Bd." volume tie.or.space.connect "volume" output.check }
  if$
  format.address.publisher.year "publisher" output.check
  pages  empty$
      'skip$            %%   evtl. hier new.sentence !!
      'part.of.sentence
   if$
  format.pages.book output
  note set.period.dash.check
  note output 
  isbn set.period.dash.check
  issn set.period.dash.check
  format.isbn.issn output
  fin.entry
}

%% geaendert 23/2/947 auch fuer Firmenschriften u."a.
FUNCTION {techreport}
{ output.bibitem
  author empty$
    { format.editors "author and editor" output.check }
    { format.authors  format.editors output.nonnull }
  if$
  set.colon.after
  format.title "title" output.check
  institution new.sentence.checka
  institution empty$
     'skip$
     { " / " institution *  output.nonnull }
  if$
  format.address.publisher.year output
  number new.sentence.checka
  format.tr.number.din "number" output.check
  new.sentence
  format.pages.book output
  note "note" output.check
  isbn set.period.dash.check
  issn set.period.dash.check
  format.isbn.issn output
  fin.entry
}

FUNCTION {default.type} { misc }

MACRO {jan} {"Januar"}

MACRO {feb} {"Februar"}

MACRO {mar} {"Maerz"}

MACRO {apr} {"April"}

MACRO {mai} {"Mai"}

MACRO {jun} {"Juni"}

MACRO {jul} {"Juli"}

MACRO {aug} {"August"}

MACRO {sep} {"September"}

MACRO {okt} {"Oktober"}

MACRO {nov} {"November"}

MACRO {dez} {"Dezember"}

%%$$$  stillgelegte Beispiele fuer den Gebrauch von Kuerzeln (hier Zs-Titel).

%%MACRO {acmcs} {"ACM Computing Surveys"}

%%MACRO {acta} {"Acta Informatica"}

%%MACRO {cacm} {"Communications of the ACM"}

%%MACRO {ibmjrd} {"IBM Journal of Research and Development"}

%%MACRO {ibmsj} {"IBM Systems Journal"}

%%MACRO {ieeese} {"IEEE Transactions on Software Engineering"}

%%MACRO {ieeetc} {"IEEE Transactions on Computers"}

%%MACRO {ieeetcad}
%% {"IEEE Transactions on Computer-Aided Design of Integrated Circuits"}

%%MACRO {ipl} {"Information Processing Letters"}

%%MACRO {jacm} {"Journal of the ACM"}

READ

FUNCTION {sortify}
{ purify$
  "l" change.case$
}

INTEGERS { len }

FUNCTION {chop.word}
{ 's :=
  'len :=
  s #1 len substring$ =
    { s len #1 + global.max$ substring$ }
    's
  if$
}

%%INTEGERS { et.al.char.used }

%%FUNCTION {initialize.et.al.char.used}
%%{ #0 'et.al.char.used :=
%%}

%%EXECUTE {initialize.et.al.char.used}

FUNCTION {format.lab.names}
{ 's :=
  s num.names$ 'numnames :=
  numnames #1 >
    { numnames #4 >
	{ #3 'namesleft := }
	{ numnames 'namesleft := }
      if$
      #1 'nameptr :=
      ""
	{ namesleft #0 > }
	{ nameptr numnames =
	    { s nameptr "{ff }{vv }{ll}{ jj}" format.name$ "others" =
		{ "\," *
%%              { "{\etalchar{+}}" *
%%                #1 'et.al.char.used :=
		}
%%              { s nameptr "{v{}}{l{}}" format.name$ * }
		{ s nameptr "{l{}}" format.name$ * }
	      if$
	    }
%%          { s nameptr "{v{}}{l{}}" format.name$ * }
	    { s nameptr "{l{}}" format.name$ * }
	  if$
	  nameptr #1 + 'nameptr :=
	  namesleft #1 - 'namesleft :=
	}
      while$
      numnames #4 >
	{ "\," *
%%      { "{\etalchar{+}}" *
%%        #1 'et.al.char.used :=
	}
	'skip$
      if$
    }
%%  { s #1 "{v{}}{l{}}" format.name$
    { s #1 "{l{}}" format.name$
      duplicate$ text.length$ #2 <
	{ pop$ s #1 "{ll}" format.name$ #3 text.prefix$ }
	'skip$
      if$
    }
  if$
}

FUNCTION {author.key.label}
{ author empty$
    { key empty$
	{ cite$ #1 #3 substring$ }
	{ key #3 text.prefix$ }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.editor.key.label}
{ author empty$
    { editor empty$
	{ key empty$
	    { cite$ #1 #3 substring$ }
	    { key #3 text.prefix$ }
	  if$
	}
	{ editor format.lab.names }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {author.key.organization.label}
{ author empty$
    { key empty$
	{ organization empty$
	    { cite$ #1 #3 substring$ }
	    { "The " #4 organization chop.word #3 text.prefix$ }
	  if$
	}
	{ key #3 text.prefix$ }
      if$
    }
    { author format.lab.names }
  if$
}

FUNCTION {editor.key.organization.label}
{ editor empty$
    { key empty$
	{ organization empty$
	    { cite$ #1 #3 substring$ }
	    { "The " #4 organization chop.word #3 text.prefix$ }
	  if$
	}
	{ key #3 text.prefix$ }
      if$
    }
    { editor format.lab.names }
  if$
}

FUNCTION {calc.label}
{ type$ "book" =
  type$ "inbook" =
  or
    'author.editor.key.label
    { type$ "proceedings" =
	'editor.key.organization.label
	{ type$ "manual" =
	    'author.key.organization.label
	    'author.key.label
	  if$
	}
      if$
    }
  if$
  duplicate$
  year field.or.null purify$ #-1 #2 substring$
  *
  'label :=
  year field.or.null purify$ #-1 #4 substring$
  *
  sortify 'sort.label :=
}

FUNCTION {sort.format.names}
{ 's :=
  #1 'nameptr :=
  ""
  s num.names$ 'numnames :=
  numnames 'namesleft :=
    { namesleft #0 > }
    { nameptr #1 >
	{ "   " * }
	'skip$
      if$
%     s nameptr "{vv{ } }{ll{ }}{  ff{ }}{  jj{ }}" format.name$ 't :=
%
%     Zeile geaendert, damit die Namenszusaetze wie von, de usw nach deutscher
%     Norm richtig einsortiert werden.   27.10.94 Lueddecke
%
      s nameptr "{ll{ }}{  ff{ }}{ vv{ }}{  jj{ }}" format.name$ 't :=
      nameptr numnames = t "others" = and
	{ "et al" * }
	{ t sortify * }
      if$
      nameptr #1 + 'nameptr :=
      namesleft #1 - 'namesleft :=
    }
  while$
}

FUNCTION {sort.format.title}
{ 't :=
  "A " #2
    "An " #3
      "Der " #4
	"Die " #4
	  "Das " #4
	    "Ein " #4
	      "Eine " #5 
		  "The " #4 t chop.word
	       chop.word
	     chop.word
	   chop.word
	 chop.word
       chop.word
    chop.word
  chop.word
  sortify
  #1 global.max$ substring$
}

FUNCTION {author.sort}
{ author empty$
    { key empty$
	{ "to sort, need author or key in " cite$ * warning$
	  ""
	}
	{ key sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.editor.sort}
{ author empty$
    { editor empty$
	{ key empty$
	    { "to sort, need author, editor, or key in " cite$ * warning$
	      ""
	    }
	    { key sortify }
	  if$
	}
	{ editor sort.format.names }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {author.organization.sort}
{ author empty$
    { organization empty$
	{ key empty$
	    { "to sort, need author, organization, or key in " cite$ * warning$
	      ""
	    }
	    { key sortify }
	  if$
	}
	{ "The " #4 organization chop.word sortify }
      if$
    }
    { author sort.format.names }
  if$
}

FUNCTION {editor.organization.sort}
{ editor empty$
    { organization empty$
	{ key empty$
	    { "to sort, need editor, organization, or key in " cite$ * warning$
	      ""
	    }
	    { key sortify }
	  if$
	}
	{ "The " #4 organization chop.word sortify }
      if$
    }
    { editor sort.format.names }
  if$
}

FUNCTION {presort}
{ calc.label
  sort.label
  "    "
  *
  type$ "book" =
  type$ "inbook" =
  or
    'author.editor.sort
    { type$ "proceedings" =
	'editor.organization.sort
	{ type$ "manual" =
	    'author.organization.sort
	    'author.sort
	  if$
	}
      if$
    }
  if$
  *
  "    "
  *
  year field.or.null sortify
  *
  "    "
  *
  title field.or.null
  sort.format.title
  *
  #1 entry.max$ substring$
  'sort.key$ :=
}

ITERATE {presort}

SORT

STRINGS { longest.label }

%% 05.03.96, Raphael Bielmeier: Ab hier geaendert,
%% Kuerzel werden im Dokument durch Zahlen ersetzt
INTEGERS { number.label longest.label.width }

FUNCTION {initialize.longest.label}
{ "" 'longest.label :=
  #1 'number.label :=
  #0 'longest.label.width :=
}

FUNCTION {longest.label.pass}
{ number.label int.to.str$ 'label :=
  number.label #1 + 'number.label :=
  label width$ longest.label.width >
    { label 'longest.label :=
      label width$ 'longest.label.width :=
    }
    'skip$
  if$
}

EXECUTE {initialize.longest.label}

ITERATE {longest.label.pass}

%% Aenderungen bis hierhin (ersetzt aus plain.bst), Raphael Bielmeier
%% 05.03.96

FUNCTION {begin.bib}
{ 
%%et.al.char.used
%%    { "\newcommand{\etalchar}[1]{$^{#1}$}" write$ newline$ }
%%    'skip$
%%  if$
  preamble$ empty$
    'skip$
    { preamble$ write$ newline$ }
  if$
  "\begin{thebibliography}{"  longest.label  * "}" * write$ newline$
}

EXECUTE {begin.bib}

EXECUTE {init.state.consts}

ITERATE {call.type$}

FUNCTION {end.bib}
{ newline$
  "\end{thebibliography}" write$ newline$
}

EXECUTE {end.bib}

%% Ende von DINALPHA.BST  KFL, 28/3/94