import java.util.*;
import java.util.Vector;
import java.io.*;
import java.util.Arrays;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.math.BigDecimal;  
import java.math.RoundingMode;


public class CRP_multi_sense{
    static int word_window;//window_size for sentence
    static int skip_gram_word_window;
    static int dimension;
    static ArrayList<String>WORD=new ArrayList<String>();// word list
    static double[]prob_word;//appearing probablity for each word, used for word omitting 
    static int[]table;
    static double MAX=5;
    static double start_learning_rate;
    static double learning_rate;
    static double Alpha=1;//initial learning rate
    static math my_math=new math();
    static double[][] vect;//global word_embedding
    static ArrayList<vocab_word>vocab=new ArrayList<vocab_word>();//vocabulary
    static Random r = new Random();
    static HashMap<Integer,HashMap<Integer,double[]>>sense_match=new HashMap<Integer,HashMap<Integer,double[]>>();
    static HashMap<Integer,HashMap<Integer,double[]>>sense_Context=new HashMap<Integer,HashMap<Integer,double[]>>();
    //sense sets for each token. each token is assocaited with a list of senses, each sense is assocaited with a embedding
    static HashMap<Integer,HashMap<Integer,Integer>>cusomers_in_table=new HashMap<Integer,HashMap<Integer,Integer>>();
    //Chinese restaurant process. each sense is assocaited with an integer indicating how many times words have been assigned
    static double gamma;// hyperparameter for Chinese restaurant process
    static String train_file="";
    static int load_embedding;
    static String embedding_file;
    static int n_doc;
    static int negative_num;
    static String save_file="";

    public static void main(String[] args)throws Exception{
        String frequency_file="";
        for(int i=0;i<args.length;i++){
            if(args[i].equals("-dimension"))
                dimension=Integer.parseInt(args[i+1]); //dimensionality 
            if(args[i].equals("-word_window"))
                word_window=Integer.parseInt(args[i+1]);//context window
            if(args[i].equals("-negative_num"))
                negative_num=Integer.parseInt(args[i+1]);//context window
            if(args[i].equals("-save_file"))
                save_file=args[i+1];//saving file
            if(args[i].equals("-frequency_file"))
                frequency_file=args[i+1];//word frequency file
            if(args[i].equals("-train_file"))
                train_file=args[i+1];//training corpus
            if(args[i].equals("-load_embedding"))
                load_embedding=Integer.parseInt(args[i+1]);//training corpus
            if(args[i].equals("-embedding_file"))
                embedding_file=args[i+1];//training corpus
            if(args[i].equals("-gamma"))
                gamma=Double.parseDouble(args[i+1]);
            if(args[i].equals("-learning_rate"))
                start_learning_rate=Double.parseDouble(args[i+1]);
        }
        ReadFre(frequency_file);// read word frequency f
        System.out.println(prob_word.length);
        vect=new double[prob_word.length][dimension];// global word embeddings
        n_doc=num_of_docs(train_file);
        random();//Initialization
        run();
        System.out.println("done");
    }

    public static void run()throws Exception{
        int thread_num=16;
        int batch_size=200;
        ExecutorService executor = Executors.newFixedThreadPool(thread_num);
        int Iter=3;
        int counter=0;
        for(int iter=0;iter<Iter;iter++){
            System.out.println(iter);
            long start_time=System.currentTimeMillis();
            java.util.Date date = new java.util.Date();
            System.out.println(date);
            ArrayList<Future<ArrayList<Integer>>>list=new ArrayList<Future<ArrayList<Integer>>>();
            //return list for multi thread
            ArrayList<ArrayList<Integer>>DocList=new ArrayList<ArrayList<Integer>>();
            ArrayList<ArrayList<Integer>>PositionList=new ArrayList<ArrayList<Integer>>();
            ArrayList<ArrayList<Integer>>PreviousSenseList=new ArrayList<ArrayList<Integer>>();
            ArrayList<Integer>Sen_Length=new ArrayList<Integer>();
            //List of Document for parallel processing
            //sense assignments for the current document from last iteration
            BufferedReader in=new BufferedReader(new FileReader(train_file));
            BufferedReader read_previous_sense=null;
            if(iter!=0)read_previous_sense=new BufferedReader(new FileReader(save_file+"store_sense"+Integer.toString(iter-1)));
            FileWriter fw_new_sense=new FileWriter(save_file+"store_sense"+Integer.toString(iter));
            //output sense labels from current iteration
            for(String line=in.readLine();line!=null;line=in.readLine()){
                counter++;
                learning_rate=start_learning_rate*(Iter*n_doc-counter)/(Iter*n_doc);
                //read a line
                String[]dict=line.split("\\s");
                String line_sense=null;
                String[]previous_senses=null;
                ArrayList<Integer>Doc=new ArrayList<Integer>();
                ArrayList<Integer>Word_Position=new ArrayList<Integer>();
                ArrayList<Integer>Previous_Senses=null;
                if(iter!=0){
                    //read sense indexes from previous iteration
                    line_sense=read_previous_sense.readLine();
                    if(line_sense==null)break;
                    previous_senses=line_sense.split("\\s");
                    Previous_Senses=new ArrayList<Integer>();
                }
                //store positions of unfiltered word
                for(int j=0;j<dict.length;j++){
                    int index=Integer.parseInt(dict[j]) ;
                    double l=1-Math.sqrt(0.0001/prob_word[index]);
                    double t=r.nextDouble();
                    if(t>l) {
                        Doc.add(index);
                        Word_Position.add(j);
                        if(iter!=0) Previous_Senses.add(Integer.parseInt(previous_senses[j]));
                    }
                    //each line corresponds to one sentence. As in Mikolov's original paper, each token has a chance to be omitted.
                }
                Sen_Length.add(dict.length);
                DocList.add(Doc);
                PositionList.add(Word_Position);
                if(iter!=0)PreviousSenseList.add(Previous_Senses);

                if(DocList.size()==batch_size){
                    int N=0;
                    for(int num=0;num<DocList.size();num++){
                        ArrayList<Integer>this_doc=DocList.get(num);
                        N+=this_doc.size();
                        ArrayList<Integer>this_position=PositionList.get(num);
                        if(this_doc.size()!=this_position.size())System.out.println("not consistent");
                        ArrayList<Integer>this_previous_sense=null;
                        int length=Sen_Length.get(num);
                        if(iter!=0)this_previous_sense=PreviousSenseList.get(num);
                        
                        Callable<ArrayList<Integer>>worker=new MyCallable_CRP(this_doc,this_position,this_previous_sense,length);
                        Future<ArrayList<Integer>>submit=executor.submit(worker);
                        list.add(submit);
                        //running each document in parallel
                    }
                    for(Future<ArrayList<Integer>>future : list){
                        ArrayList<Integer>new_sense_list=future.get();
                        for(int sense_index:new_sense_list)
                            fw_new_sense.write(Integer.toString(sense_index)+" ");
                        fw_new_sense.write("\n");
                    }
                    //System.out.println(vect[0][0]);
                    //System.out.println(N/(1.0*(end_time-start_time)/1000));
                    //System.out.println(vect[0][0]);
                    list=new ArrayList<Future<ArrayList<Integer>>>();
                    DocList=new ArrayList<ArrayList<Integer>>();

                    PositionList=new ArrayList<ArrayList<Integer>>();
                    Sen_Length=new ArrayList<Integer>();
                    PreviousSenseList=null;
                    if(iter!=0) PreviousSenseList=new ArrayList<ArrayList<Integer>>();
                }
            }
            fw_new_sense.close();
            long end_time=System.currentTimeMillis();
            System.out.println((end_time-start_time)/1000);
        }
        Save(save_file);
        executor.shutdown();
        System.out.println("Chinese Restaurant Processes Done");
    }

    public static class MyCallable_CRP implements Callable<ArrayList<Integer>>{
        ArrayList<Integer>sen=null;
        ArrayList<Integer>position=null;
        ArrayList<Integer>previous_senses=null;
        int total_length=-1000;
        public MyCallable_CRP(ArrayList<Integer>doc,ArrayList<Integer>position,ArrayList<Integer>previous_senses,int total_length){
            this.sen=doc;
            this.position=position;
            this.previous_senses=previous_senses;
            this.total_length=total_length;
        }

        public ArrayList<Integer> call()throws Exception {
            ArrayList<Integer>NewSenseList=CRP();
            return NewSenseList;
        }
        public ArrayList<Integer> CRP(){
            ArrayList<Integer>NewSenseList=new ArrayList<Integer>();
            if(sen.size()==0){
                for(int ik=0;ik<total_length;ik++)
                    NewSenseList.add(-1);
                return NewSenseList;
            }
            int LL=-100;
            for(int i=0;i<sen.size();i++){
                try{
                int current_position_original=position.get(i);
                int previous_position;

                if(i==0)previous_position=0;
                else previous_position=position.get(i-1)+1;
                for(int ik=previous_position;ik<current_position_original;ik++)
                    NewSenseList.add(-1);

                int word=sen.get(i);
                int previous_sense_index=-1000;
                if(previous_senses!=null)previous_sense_index=previous_senses.get(i);
                int half=1+(int)(word_window/2*r.nextDouble());
                int num_collected=0;
                double[]context_v;
                context_v=new double[dimension];
                HashMap<Integer,double[]>sense_List=sense_match.get(word); //sense list for current token
                HashMap<Integer,double[]>sense_context=sense_Context.get(word);
                HashMap<Integer,Integer>sense_table=cusomers_in_table.get(word);

                if(previous_sense_index!=-1000&&previous_sense_index!=-1)
                    sense_table.put(previous_sense_index,sense_table.get(previous_sense_index)-1);

                ArrayList<Integer>Neighs=new ArrayList<Integer>();
                for(int position=i-1;position>=i-half;position--){
                    if(position<0)break;
                    int n_index=sen.get(position);
                    Neighs.add(n_index);
                }
                for(int position=i+1;position<=i+half;position++){
                    if(position>=sen.size())break;
                    int n_index=sen.get(position);
                    Neighs.add(n_index);
                }
                int num_neigh=Neighs.size();
                if(num_neigh==0){
                    NewSenseList.add(-1);
                    continue;
                }
                for(int n_index:Neighs){
                    context_v=my_math.plus(context_v,vect[n_index]);
                }
                context_v=my_math.dot(1.00/num_neigh,context_v);
                //obtain context vector
                //Chinese Restaurant Process
                int should_new=-1; //whether we should set up a new sense
                double[]prob;
                if(sense_List.size()==0){
                    sense_List.put(0,vect[word].clone());
                    sense_table.put(0,1);
                    sense_context.put(0,context_v.clone());
                    NewSenseList.add(0);
                    continue;
                }

                if(sense_List.size()<20){
                        // maximum 20 senses for each word
                    prob=new double[sense_List.size()+1];
                    prob[sense_List.size()]=gamma;
                    should_new=1;
                }
                else {
                    // if more than 20 senses, we should not set up a new sense
                    prob=new double[sense_List.size()];
                    should_new=0;
                }
                for(int j=0;j<prob.length-1;j++){
                        //iterating over each sense, computing probablity assigning current token to each sense     
                    double[]sense_context_v;
                    sense_context_v=sense_context.get(j);
                    prob[j]=sense_table.get(j)*my_math.sigmod(my_math.dot(context_v,sense_context_v));
                }
                for(int j=1;j<prob.length;j++){
                    prob[j]=prob[j-1]+prob[j];
                }
                double sample=r.nextDouble()*prob[prob.length-1];
                //sampling sense label from Chinese restaurant problem
                int new_label=-1;
                for(int j=0;j<prob.length;j++){
                    if(sample<prob[j]){
                        new_label=j;
                        break;
                    }
                }
                NewSenseList.add(new_label);
                LL=new_label;
                
                if (new_label==-1){
                    System.out.println("lable-1");
                    System.out.println("context_v");
                    System.out.println(context_v[0]);
                    //System.out.println("sense_context_v");
                    //System.out.println(sense_context_v[0]);
                    printDoubleArray(prob);
                }

                    //update parameters involved, both sense embeddings and gloabl word embeddings
                if((new_label!=prob.length-1&&should_new==1)||should_new==0){
                    double[]sense_context_v;
                    sense_context_v=sense_context.get(new_label);
                    sense_context_v=my_math.plus(sense_context_v,my_math.dot(Alpha,context_v));
                    sense_table.put(new_label,sense_table.get(new_label)+1);
                }
                else{
                    sense_context.put(prob.length-1,context_v.clone());
                    sense_table.put(prob.length-1,1);
                    sense_List.put(prob.length-1,vect[word].clone());
                }
                double[]current_sense_emb;
                //current_sense_emb=sense_List.get(new_label);
                double f;double g=0;
                current_sense_emb=sense_List.get(new_label);
                
                double[]neu1e;neu1e=new double[dimension];
                for(int neigh_position=0;neigh_position<Neighs.size();neigh_position++){
                    int l1=Neighs.get(neigh_position);
                    f=0;
                    for(int k=0;k<dimension;k++) f+=current_sense_emb[k]*vect[l1][k];
                    if(f>MAX||f<-MAX)g=0;
                    else g=(1-my_math.sigmod(f))*learning_rate;
                    for(int k=0;k<dimension;k++){
                        neu1e[k]+=g*vect[l1][k];
                        vect[l1][k]+=g*current_sense_emb[k];
                    }

                    for(int neg=0;neg<negative_num;neg++){
                        int l2=table[r.nextInt(table.length)];
                        if(l2==l1)continue;
                        f=0;
                        for(int k=0;k<dimension;k++) f+=current_sense_emb[k]*vect[l2][k];
                        if(f>MAX||f<-MAX) continue;
                        g=-my_math.sigmod(f)*learning_rate;
                        for(int k=0;k<dimension;k++){
                            neu1e[k]+=g*vect[l2][k];
                            vect[l2][k]+=g*current_sense_emb[k];
                        }
                    }
                }
                for(int k=0;k<dimension;k++) current_sense_emb[k]+=neu1e[k];
                }
                catch(Exception e){
                    NewSenseList.add(-1);
                    System.out.println("problem");
                    continue;
                }
            }
            if(position.size()==0)System.out.println("empty sentence");
            if(position.get(position.size()-1)!=total_length-1){
                for(int ik=position.get(position.size()-1)+1;ik<total_length;ik++)
                    NewSenseList.add(-1);
            }
            if(total_length!=NewSenseList.size()){
                System.out.println("wait wait problem");
                System.out.println(LL);
                printArrayList(sen);
                printArrayList(position);
                System.out.println(total_length);
                printArrayList(NewSenseList);
            }
            return NewSenseList;
        }
    }

    public static void random(){
        //random initialization
        double epso=0.1;
        for(int i=0;i<vect.length;i++){
            for(int j=0;j<dimension;j++){
                vect[i][j]=(r.nextDouble()*2*epso-epso);
            }
        }
    }


    public static void Save(String filename) throws Exception{
        //save file
        //save sense specific embeddings
        System.out.println("start saving");
        FileWriter fw=new FileWriter(filename+"_vect_sense");
        FileWriter fw1=new FileWriter(filename+"_vect_sense_context");
        for(int i=0;i<vect.length;i++){
            HashMap<Integer,double[]>match=sense_match.get(i);
            HashMap<Integer,Integer>table=cusomers_in_table.get(i);
            HashMap<Integer,double[]>sense_context=sense_Context.get(i);
            int number_of_senses=-1;
            if(match.size()!=table.size()){
                System.out.println("size not consistent");
                System.out.println(match.size());
                System.out.println(table.size());
                if(match.size()<table.size())number_of_senses=match.size();
                else number_of_senses=table.size();
            }
            else number_of_senses=match.size();
            //if(match.size()!=1&&match.size()!=0 ){
            if(match.size()!=0){
                fw.write("word "+Integer.toString(i)+"\n");
                fw1.write("word "+Integer.toString(i)+"\n");
                int count=-1;
                int total_num=0;
                for(int index=0;index<number_of_senses;index++)
                    total_num=total_num+table.get(index);
                for(int index=0;index<number_of_senses;index++){
                    count++;
                    //ignore senses with less than occuring chance of 0.01
                    if (1.0*table.get(index)/total_num<0.01)continue;
                    fw.write("sense"+Integer.toString(count)+" "+String.valueOf(1.0*table.get(index)/total_num )+"\n");
                    fw1.write("sense"+Integer.toString(count)+"\n");
                    double[]store;
                    store=match.get(index);
                    double[]context;
                    context=sense_context.get(index);
                    if(store==null){
                        System.out.println("null problem");
                        continue;
                    }
                    for(int k=0;k<dimension;k++){
                        if(k!=dimension-1){
                            fw.write(store[k]+" ");
                            fw1.write(context[k]+" ");
                        }
                        else{
                            fw.write(store[k]+"\n");
                            fw1.write(context[k]+"\n");
                        }
                    }
                }
            }
        }
        fw.close();
        fw1.close();
            //save global embeddings
        fw=new FileWriter(filename+"_vect_global");
        for(int i=0;i<vect.length;i++){
            for(int j=0;j<vect[0].length;j++){
                if(j!=vect[0].length-1)
                    fw.write(vect[i][j]+",");
                else fw.write(vect[i][j]+"\n");
            }
        }
        fw.close();
    }
    
    public static int num_of_docs(String filename)throws IOException {
    // compute number of docs
        BufferedReader in=new BufferedReader(new FileReader(filename));
        int n_line=0;
        for(String line=in.readLine();line!=null;line=in.readLine())
            n_line++;
        return n_line;
    }

    public static void readvect(String read_vect_file)throws IOException{
        BufferedReader in=new BufferedReader(new FileReader(read_vect_file));
        int counter=-1;
        for(String line=in.readLine();line!=null;line=in.readLine()){
            counter++;
            String[]string_split=line.trim().split(","); 
            for(int i=0;i<string_split.length;i++)
                vect[counter][i]=Double.parseDouble(string_split[i]);
        }
    }

    public static int num_of_lines(String filename)throws IOException {
        //compute number of lines
        BufferedReader in=new BufferedReader(new FileReader(filename));
        int n_line=0;
        for(String line=in.readLine();line!=null;line=in.readLine()){
            n_line++;
        }
        in.close();
        return n_line;
    }

    public static void ReadFre(String filename)throws IOException {
    //read word frequency
        BufferedReader in=new BufferedReader(new FileReader(filename));
        int i=-1;
        int total=0;
        int word_num=0;
        for(String line=in.readLine();line!=null;line=in.readLine()){
            word_num++;
        }
        //System.out.println(filename);
        //System.out.println(word_num);
        prob_word=new double[word_num];
        in=new BufferedReader(new FileReader(filename));
        for(String line=in.readLine();line!=null;line=in.readLine()){
            i++;
            String[]dict=line.split("\\s");
            prob_word[i]=Double.parseDouble(dict[0]);
        }
        for(i=0;i<word_num;i++){
            HashMap<Integer,double[]>t_1=new HashMap<Integer,double[]>();
            sense_match.put(i,t_1);
            HashMap<Integer,double[]>t_3=new HashMap<Integer,double[]>();
            sense_Context.put(i,t_3);
            HashMap<Integer,Integer>t_2=new HashMap<Integer,Integer>();
            cusomers_in_table.put(i,t_2);
        }
        table=new int[100000000];
        double train_words_pow=0;
        for(i=0;i<word_num;i++)
            train_words_pow+=Math.pow(prob_word[i],0.75);
        i=0;double dl=Math.pow(prob_word[i],0.75)/train_words_pow;
        for(int a=0;a<table.length;a++){
            table[a]=i;
            if(1.0*a/table.length>dl){
                i++;
                dl+=Math.pow(prob_word[i],0.75)/train_words_pow;
            }
            if(i>=prob_word.length-1)i=prob_word.length-1;
        }
    }

    public static void printArrayList(ArrayList<Integer>A){
        String string="";
        for(int i=0;i<A.size();i++)
            string=string+Integer.toString(A.get(i))+" ";
        System.out.println(string);
    }
    public static void printArray(int[]A){
        String string="";
        for(int i=0;i<A.length;i++)
            string=string+Integer.toString(A[i])+" ";
        System.out.println(string);
    }
    public static void printDoubleArray(double[]A){
        String string="";
        for(int i=0;i<A.length;i++)
            string=string+Double.toString(A[i])+" ";
        System.out.println(string);
    }
}

class vocab_word{
    ArrayList<Integer>point=new ArrayList<Integer>();
    ArrayList<Integer>code=new ArrayList<Integer>();
}
