#The project involves data preprocessing, preparation, nonlinear statistical model fitting & validation using 
#maximum likelihood ratio test. The project aims at business use case creation & actionable recommendations to 
#increase the IT contract renewal conversion rate using R programming.

models<-function(par,x1,x2,x3,y1,temp, par_def, par_int, int_ctr, var_pos ) {
  #dim returns
  dr = var_pos[1]+var_pos[2]+var_pos[3]+1
  
  #create par[] text values for formula creation
  temp_text = temp
  ptr1=1  
  for (t in 1:length(temp_text)){
    if (temp_text[t]==1){
      temp_text[t]= paste('par[',ptr1,']', sep = '')
      ptr1 = ptr1 + 1
    }
  }
  
  
  token=c()       # Generate individual tokens and store in token
  token[1]=paste('SSE<-sum((','y1')
  token_ctr=2
  ptr=1 
  # temp[1]=1; par_def[1]=0
  
  # Generate Intercept Variable
  for (i in 1:var_pos[1]){
    if (temp[i]==1){
      token[token_ctr]= paste('-',temp_text[i], sep = '')   #paste('-par[',ptr,']', sep = '')
      #ptr = ptr + 1
      token_ctr = token_ctr + 1 
    } else {
      if (temp[i] == 0 && par_def[i] != 0){ 
        token[token_ctr]=paste('-',par_def[i], sep = '')
        #ptr = ptr + 1
        token_ctr = token_ctr + 1 }
    }
  }
  
  # Generate Independant variables
  result=c()
  x=1
  #token_ctr = 3; ptr=2; dr= 5
  #temp[6]=1;  temp[2]=0;temp[3]=0;par_def[2]=.5; par_def[3]=.6 ; par_def[5]=.05 
  for (i in (var_pos[1]+1):(var_pos[1]+var_pos[2])){
    if (temp[i]==1){
      result[x] = paste ('if (',temp_text[dr],'==0) { result',x,' <- x',x,' }  else { result',x,' <- ((1-exp(-',temp_text[dr],'*x',x,'))/(',temp_text[dr],')) }', sep = '')
      token[token_ctr] = paste('-(',temp_text[i],'*(', result[x],'))', sep = '')
      eval(parse(text=result[x]))
      
      #ptr = ptr + 1
      token_ctr = token_ctr + 1 
      dr = dr + 1 
      x = x + 1 
    } else {
      if (temp[i] == 0 && par_def[i] != 0){ 
        
        result[x] = paste ('if (',par_def[dr],'==0) { result',x,' <- x',x,' }  else { result',x,' <- ((1-exp(-',par_def[dr],'*x',x,'))/',par_def[dr],') }', sep = '')
        
        token[token_ctr] = paste('-(',par_def[i],'*(',result[x],'))', sep = '')
        
        eval(parse(text=result[x]))
        
        #ptr = ptr + 1
        token_ctr = token_ctr + 1 
        dr = dr + 1
        x = x + 1 }
    }
  }
  
  # Generate Interaction variables
  
  #int_ctr =0; i=1
  for (i in (var_pos[1]+var_pos[2]+1):(var_pos[1]+var_pos[2]+var_pos[3])){
    
    
    if (temp[i]==1){
      
      coeff=c()
      int_p=1
      for(j in 1:length(par_int[i-var_pos[1]-var_pos[2],])){
        if(par_int[i-var_pos[1]-var_pos[2], j]==1) {
          coeff[int_p] = colnames(par_int[i-var_pos[1]-var_pos[2],][j])
          int_p = int_p + 1 }
      }
      
      
      result[x] = paste ('result',x,' <- ', paste(coeff,collapse = '*'), sep = '')
      
      token[token_ctr] = paste('-(',temp_text[i],'*(', result[x],'))', sep = '')
      #int_ctr=int_ctr+1 
      #ptr = ptr + 1
      token_ctr = token_ctr + 1
      x = x + 1
      
    } else { if (par_def[i] != 0) {
      
      coeff=c()
      int_p=1
      for(j in 1:length(par_int[i-var_pos[1]-var_pos[2],])){
        if(par_int[i-var_pos[1]-var_pos[2], j]==1) {
          coeff[int_p] = colnames(par_int[i-var_pos[1]-var_pos[2],][j])
          int_p = int_p + 1 }
      }
      
      result[x] = paste ('result',x,' <- ', paste(coeff,collapse = '*'), sep = '')
      
      token[token_ctr] = paste('-(',par_def[i],'*(',result[x],'))', sep = '')
      #int_ctr=int_ctr+1 
      #ptr = ptr + 1
      token_ctr = token_ctr + 1
      x = x + 1
      }
    }
    
    eval(parse(text=result[x]))
    
  }
  
  text1 = paste(paste(token, sep = ' ', collapse = ''), ')^2)', sep = '')
  
  eval(parse(text=text1))
  
}


HA_function<-function(fun,x1,x2,x3,y1, temp, par_def, par_int, int_ctr1, var_pos){
  
  temp_len <- as.numeric(rowSums(temp[1,] == "1"))
  par<-rep.int(0, temp_len)
  #par<-c(0,0,0,0,0)
  coef = c()
  A_SSE = NULL
  #################
  ###### change models to fun
  sserr<-nlminb(par,fun,x1=x1,x2=x2,x3=x3,y1=y1, temp=temp, par_def= par_def, par_int = par_int, int_ctr = int_ctr1, var_pos = var_pos)
  #################
  temp_result = temp #par_res[1,]
  ptr1=1
  for (t in 1:length(temp_result)){
    if (temp_result[t] ==1){
      temp_result[t]= sserr$par[ptr1]
      ptr1 = ptr1 + 1
    }
  }
  
  A_SSE<-sserr$objective
  coef = cbind(temp_result,A_SSE)
}


H0_function<-function(fun,x1,x2,x3,y1, temp, par_def, par_int, int_ctr1, var_pos){
  
  BS_SSE<-data.frame()
  
  #Bootstrap Null function
  
  temp_len <- as.numeric(rowSums(temp[1,] == "1"))
  par<-rep.int(0, temp_len)
  
  nb = 200
  BSSE = NULL
  n = NROW(y1)
  
  for (j in 1:nb){
    print(j)
    unifnum = sample(c(1:n),n,replace = T)	# pick random indices
    
    y_samp<-y1[unifnum,]
    x1_samp<-x1[unifnum,]
    x2_samp<-x2[unifnum,]
    x3_samp<-x3[unifnum,]
    
    Boot_SSE<-nlminb(par,fun,x1=x1_samp,x2=x2_samp,x3=x3_samp,y1=y_samp, temp=temp, par_def= par_def, par_int = par_int, int_ctr = int_ctr1, var_pos = var_pos)
    
    BSSE[j]= Boot_SSE$objective
    
  }
  BS_SSE = rbind(BS_SSE,t(BSSE))
}

###################################################################################################################

test<-function(m1,m2,data,p_res,p_int,p_def,p_pos){
  
  #Main Loop to calculate different models
  
  
  input<-read.csv(data, header = TRUE, sep = ',')
  x1 = input[1]
  x2 = input[2]
  x3 = input[3]
  y=input[4]
  #y<-20+2*((1-exp(-0.1*x1))/0.1)+4*((1-exp(-0.5*x2))/0.5)+6*x1*x2+e
  
  par_res<-read.csv(p_res, header = TRUE, sep = ',')
  
  par_int<-read.csv(p_int, header = TRUE, sep = ',')
  
  par_def<-p_def
  
  var_pos<-p_pos
  
  
  Coef_DF<-data.frame()
  
  BS_SSE_main<-data.frame()
  
  TEST_STAT = NULL
  
  HA= m1
  H0= m2
  
  int_ctr_HA = 0
  
  int_ctr_H0 = 0

  
  print('ALT Hypo Evaluation')
  temp = par_res[HA,]
  
  HA_function_coef<-HA_function(models,x1=x1,x2=x2,x3=x3,y1=y, temp=temp, par_def=par_def, par_int=par_int, int_ctr1=int_ctr_HA, var_pos=var_pos)
  
  print(HA_function_coef)
  
  print('NULL Hypo Evaluation')
  temp1 = par_res[H0,]
  
  H0_function_coef<-H0_function(models,x1=x1,x2=x2,x3=x3,y1=y, temp=temp1 , par_def=par_def, par_int=par_int, int_ctr1=int_ctr_H0, var_pos=var_pos)
  
  print(H0_function_coef)
  
  #ALT_SSE=HA_function_coef[length(HA_function_coef)]
  Coef_DF = rbind(Coef_DF,data.frame(HA_function_coef))
  BS_SSE_main = rbind(BS_SSE_main, H0_function_coef)
  
  print('Test Statistic')
  
  TEST_STAT = mean(HA_function_coef[1,length(HA_function_coef)]>H0_function_coef[1,])
  
  print(TEST_STAT)
  

  #Create final dataframe having model coef, alt sse value and test statistic
  Datafile = data.frame(Coef_DF, TEST_STAT)
  write.csv(Datafile, file="Test_Results.csv")

}

test(24,6, 'project1.1.csv', 'restrictions.csv', 'interactions.csv' ,c(0,0,0,0,0,0,0,0,0,0,0), c(1,3,4,3))
