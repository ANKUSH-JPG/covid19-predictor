#!/usr/bin/env python
# coding: utf-8

# In[1]:


import smtplib


# In[ ]:





# In[8]:


file=open('accuracy.txt','r')


# In[9]:


accuracy=file.read()
file.close()


# In[11]:


Text="THE ACCURACY OF THE MODEL IS"

message=Text+" "+accuracy



# In[3]:


s=smtplib.SMTP(host='smtp.gmail.com',port=587)


# In[4]:


s.starttls()


# In[13]:


s.login("forallchallenge@gmail.com","Ankush@12345")

# In[15]:


s.sendmail("forallchallenge@gmail.com","forallchallenge@gmail.com",message)
s.quit()


# In[ ]:




