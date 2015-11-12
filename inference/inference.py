import sys,math
import numpy,re,random
#the code takes as inputs sense-specific embeddings, and lines of tokens and assign sense specific embedding to each token within the line

def ReadFrequency(fre_file):
    A=open(fre_file,"r");
    lines=A.readlines();
    Frequency={};
    for i in range(len(lines)):
        Frequency[i]=float(lines[i]);
    return Frequency;


def load_embeddings(Global_file,Sense_file):
    # load embeddings
    A1=open(Global_file,"r");
    lines1=A1.readlines();
    vect_global={};
    vect_multi={};
    for i in range(len(lines1)):
        vect_global[i]=numpy.array(lines1[i].split(","),dtype=float);
    A1=open(Sense_file,"r");
    for line in A1:
        if line.find("word")!=-1:
            t1=line.find(" ");
            t2=line.find(" ",t1+1);
            w_index=int(line[t1:t2]);# extract word index
            vect_multi[w_index]=[];
        if line.find("sense")!=-1: # extract current sense index
            t1=line.find("sense");
            t2=line.find(" ",t1+1);
            current_sense=line[t1:t2].strip(); #current sense index
            current_prob=float(line[t2:].strip()); #chinese restaurant probability for current word
        if line.find("word")==-1 and line.find("sense")==-1:
            vect_multi[w_index].append((current_prob,numpy.array(line.strip().split(" "),dtype=float)));
            #tuple, [probability,vector]
    return [vect_global,vect_multi];

def Calculate(input_file,output_file,vect_global,vect_sense,Frequency,isGreedy):
    # pick up the sense embedding with the largest likelihood
    A=open(input_file,"r");
    B=open(output_file,"w");
    for line in A:
        word_list=line.strip().split(" ");
        for i in range(len(word_list)):
            current_word=int(word_list[i]);
            context_v=get_context_vect(i,word_list,vect_global,Frequency);
            if vect_sense.has_key(current_word)is False:
                current_vect=vect_global[current_word];
            else:
                if isGreedy==1:
                    #Greedy sense embedding, pick up the sense specific embedding with largest probability given context
                    #calculate probabilty of each sense given context with Chinese restaurant processes
                    sense_list=vect_sense[current_word];
                    Prob=len(sense_list)*[1];
                    for i in range(len(sense_list)):
                        sense_vect=sense_list[i][1];
                        Prob[i]=sense_list[i][0]*Prob[i]/(1+numpy.exp(-numpy.dot(sense_vect,context_v)));
                    index=Prob.index(max(Prob));
                    current_vect=sense_list[index][1];
                else:
                    #Expectation
                    current_vect=numpy.zeros(vect_global[0].shape);           
                    sense_list=vect_sense[current_word];
                    #calculate probabilty of each sense given context with Chinese restaurant processes
                    Sum_P=0;
                    for i in range(len(sense_list)):
                        sense_vect=sense_list[i][1];
                        P=sense_list[i][0]/(1+numpy.exp(-numpy.dot(sense_vect,context_v)));
                        Sum_P=Sum_P+P;
                        current_vect=current_vect+P*sense_vect;
                    current_vect=current_vect/Sum_P;
                    
            for value in current_vect:
                B.write(str(value)+" ");
            B.write("\n");
        B.write("\n");
            
def get_context_vect(index,sen,Global_vect,Frequency):
    #get context vector for current word
    context_v=numpy.zeros(Global_vect[0].shape);
    current_index=index;
    num_select=0;
    neigh_list=[];
    # collect neighboring words and achieve context vector
    while 1==1:
        current_index=current_index-1;
        if current_index<0:break;
        word=int(sen[current_index]);
        if Global_vect.has_key(word)is False:continue;
        l=1-math.sqrt(0.0001/Frequency[word]);
        t=random.random();
        if(t<l):continue;
        # omit frequent word
        neigh_list.append(word);
        num_select=num_select+1;
        context_v=numpy.add(context_v,Global_vect[word]);
        if num_select==7:break;
    current_index=index;
    num_select=0;
    while 1==1:
        current_index=current_index+1;
        if current_index>=len(sen):break;
        word=int(sen[current_index]);
        if Global_vect.has_key(word)is False:continue;
        l=1-math.sqrt(0.0001/Frequency[word]);
        t=random.random();
        if(t<l):continue;
        # omit frequent word
        neigh_list.append(word);
        num_select=num_select+1;
        context_v=numpy.add(context_v,Global_vect[word]);
        if num_select==7:break;
    context_v=context_v/len(neigh_list);
    return context_v;  

for i in range(0,len(sys.argv)):
    if sys.argv[i]=="-global_file":
        Global_file=sys.argv[i+1];
    if sys.argv[i]=="-sense_file":
        Sense_file=sys.argv[i+1];
    if sys.argv[i]=="frequency_file":
        Frequency_file=sys.argv[i+1];
    if sys.argv[i]=="-input_file":
        input_file=sys.argv[i+1];
    if sys.argv[i]=="-output_file":
        output_file=sys.argv[i+1];
    if sys.argv[i]=="-isGreedy":
        isGreedy=int(sys.argv[i+1]);       

[vect_global,vect_multi]=load_embeddings(Global_file,Sense_file);
Frequency=ReadFrequency(Frequency_file);

# if isGreedy is 1, use greedy sense embeddings. Otherwise use expectation
Calculate(input_file,output_file,vect_global,vect_multi,Frequency,isGreedy)
