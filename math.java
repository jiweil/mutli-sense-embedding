public class math{
    static double[]table;
    static double[]tanh_table;
    static int num=0;
    math(){
        table=new double[6000];
        for(int i=0;i<6000;i++){
            table[i]=Math.exp(0.001*i-3);
        }
        tanh_table=new double[8000];
        for(int i=0;i<8000;i++){
            double c=0.001*i-4;
            double a1=Math.exp(-c);
            double a2=Math.exp(c);
            tanh_table[i]=(a2-a1)/(a2+a1);
        }
    }
    public double exp(double value){
        if(value>-3&&value<3){
            return table[(int)(value/0.001)+3000];
        }
        else{
            //System.out.println("warning");
            //System.out.println(value);
            return Math.exp(value);
        }
    }
    public double L1(double []a1){
        double total=0;
        for(int i=0;i<a1.length;i++)
            total=total+a1[i]*a1[i];
        return Math.sqrt(total);
    }

    public double dot(double[]a1,double[]a2){
        double total=0;
        for(int i=0;i<a1.length;i++)
            total+=a1[i]*a2[i];
        return total;
    }
    public double[][] vector_vector_M(double[]a1,double[]a2){
        double[][]G;G=new double[a1.length][a2.length];
        for(int i=0;i<a1.length;i++)
            for(int j=0;j<a2.length;j++)
                G[i][j]=a1[i]*a2[j];
        return G;
    }

    public double tanh(double a){
        /*
        if(a>-4&&a<4)
            return tanh_table[(int)(a/0.001)+4000];
        else{
            double a1=Math.exp(-a);
            double a2=Math.exp(a);
            return (a2-a1)/(a2+a1);
        }
        */

            double a1=Math.exp(-a);
            double a2=Math.exp(a);
            return (a2-a1)/(a2+a1);

    }
    public double derivative_tanh(double a){
        return 1-Math.pow(a,2);
    }

    public double[]derivative_tanh(double[]a){
        double[]b;b=new double[a.length];
        for(int i=0;i<a.length;i++)b[i]=derivative_tanh(a[i]);
        return b;
    }

    public double[][][]Vatrix_Matrix_Dot(double[][][]A,double[][]B){
        if(A[0][0].length!=B[0].length)System.out.println("Matrix Dimensions not Constant");
        double[][][]C;C=new double[A.length][A[0].length][B.length];
        for(int i=0;i<A.length;i++){
            for(int j=0;j<A[i].length;j++){
                for(int k=0;k<B.length;k++){
                    for(int m=0;m<A[i][j].length;m++){
                        C[i][j][k]+=A[i][j][m]*B[k][m];
                    }
                }
            }
        }
        return C;
    }


    public double sigmod(double a){
        return 1/(1+Math.exp(-a));
    }
    public double derivative_sigmod(double a){
        return a*(1-a);
    }
    
    public double[]tanh(double []a1){
        double[]A;A=new double[a1.length];
        for(int i=0;i<a1.length;i++)
            A[i]=tanh(a1[i]);
        return A;
    }
    public double[][]dot(double[][]a1,double[][]a2){
        double[][]A;
        A=new double[a1.length][a2.length];
        for(int i=0;i<a1.length;i++)
            for(int j=0;j<a2.length;j++)
                for(int k=0;k<a1[0].length;k++)
                    A[i][j]+=a1[i][k]*a2[j][k];
        return A;
    }

    public double[]dot(double[][] a1,double[] a2){
        double []A;
        A=new double[a1.length];
        for(int i=0;i<a1.length;i++){
            A[i]=0;
            for(int k=0;k<a1[0].length;k++)
                A[i]+=a1[i][k]*a2[k];
        }
        return A;
    }
    public double[]plus(double[]a1,double[]a2){
        double []c;c=new double[a1.length];
        for(int i=0;i<a1.length;i++)
            c[i]=a1[i]+a2[i];
        return c;
    }

    public double[] Self_Plus(double[]a1,double[]a2){
        for(int i=0;i<a1.length;i++)
            a1[i]=a1[i]+a2[i];
        return a1;
    }

    public double[] dot_dot(double[]a1,double[]a2){
        double[]c;c=new double[a1.length];
        for(int i=0;i<a1.length;i++)
            c[i]=a1[i]*a2[i];
        return c;
    }
    public double[]dot(double a1,double[]a2){
        double[]c;c=new double[a2.length];
        for(int i=0;i<a2.length;i++)
            c[i]=a1*a2[i];
        return c;
    }
    public double[][]dot(double a1,double[][]a2){
        double[][]c;c=new double[a2.length][a2[0].length];
        for(int i=0;i<a2.length;i++)
            for(int j=0;j<a2[0].length;j++)
                c[i][j]=a1*a2[i][j];
        return c;
    }
    public double AverageArray(double []a){
        double sum=0;
        for(int i=0;i<a.length;i++)
            sum+=a[i];
        return sum/a.length;
    }

    public double Sum(double []a){
        double sum=0;
        for(int i=0;i<a.length;i++)
            sum+=a[i];
        return sum;
    }
    public double[] Matrix_Line_Sum(double[][]a){
        double[]L;
        L=new double[a.length];
        for(int i=0;i<a.length;i++)
            for(int j=0;j<a[0].length;j++)
                L[i]+=a[i][j];
        return L;
    }

    public double[]copy(double[]a2){
        double[]a1;
        a1=new double[a2.length];
        for(int i=0;i<a1.length;i++)
            a1[i]=a2[i];
        return a1;
    }
    public double[]copy(double[]a2,int begin,int end){
        double[]a1;a1=new double[end-begin];
        for(int i=begin;i<end;i++)
            a1[i-begin]=a2[i];
        return a1;
    }
    public double[][]copy(double[][]a2){
        double[][]a1;a1=new double[a2.length][a2[0].length];
        for(int i=0;i<a1.length;i++){
            for(int j=0;j<a1[0].length;j++)
                a1[i][j]=a2[i][j];
        }
        return a1;
    }
    public double[][]Matrix_Transpose(double[][]a2){
        double[][]a1;a1=new double[a2[0].length][a2.length];
        for(int i=0;i<a2.length;i++)
            for(int j=0;j<a2[0].length;j++)
                a1[j][i]=a2[i][j];
        return a1;
    }
    public double[][]copy(double[][]a2,int row_begin,int row_end,int column_begin,int column_end){
        double[][]a1;a1=new double[row_end-row_begin][column_end-column_begin];
        for(int i=row_begin;i<row_end;i++)
            for(int j=column_begin;j<column_end;j++)
                a1[i-row_begin][j-column_begin]=a2[i][j];
        return a1;
    }
}
