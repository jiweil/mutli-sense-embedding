import sys,re;
# input list of files, output dictionary 
def GetFreList(filename):
    #get word frequency list for words in the input file
    Frequency_List={};
    A=open(filename,'r');
    N_word=0;
    line_num=0;
    for line in A:
        line_num=line_num+1;
        G=re.split(" |\t",line.strip().lower());
        if len(G)==0:continue;
        N_word=N_word+len(G);
        for item in G:
            if len(item)==0:continue;
            if Frequency_List.has_key(item)is False:
                Frequency_List[item]=0;
            Frequency_List[item]=Frequency_List[item]+1;
    return [Frequency_List,N_word];
    
def getDic(Frequency_List,vocab_size,dicFile,frequency_file,N_word):
    sort=sorted(Frequency_List.items(), key=lambda x: x[1],reverse=True)
    Dic={};
    Dic['Unknown']=0;
    write_dic=open(dicFile,"w");
    write_dic.write('UNknown\n');
    write_frequency=open(frequency_file,"w");
    if len(sort)<vocab_size-1:
        V=len(sort);
        write_frequency.write("0\n");
    else:
        V=vocab_size;
        total_unknown=0;
        for i in range(V-1,len(Frequency_List)):
            word=sort[i][0];
            total_unknown=total_unknown+Frequency_List[word];
        write_frequency.write(str(1.00*total_unknown/N_word)+"\n");

    for i in range(V-1):
        word=sort[i][0];
        Dic[word]=i+1;
        write_dic.write(word+"\n")
        write_frequency.write(str(sort[i][1]*1.00/N_word)+"\n");
    return Dic;
def GetFileIndex(Dic,inputfile,outputfile):
    input_=open(inputfile,"r");
    output_=open(outputfile,"w");
    for line in input_:
        word_list=line.strip().split(" ");
        for item in word_list:
            if len(item)==0:continue;
            if Dic.has_key(item):
                output_.write(str(Dic[item])+" ");
            else: output_.write("0 ");
        output_.write("\n");
    
vocab_size=int(sys.argv[1]);#vocab size
dicFile=sys.argv[2];#store dictionary
frequency_file=sys.argv[3];
outputfile=sys.argv[4];
inputfile=sys.argv[5];
[Frequency_List,N_word]=GetFreList(inputfile);
Dic=getDic(Frequency_List,vocab_size,dicFile,frequency_file,N_word);
GetFileIndex(Dic,inputfile,outputfile);

