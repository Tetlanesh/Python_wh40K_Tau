import numpy as np   
import pandas as pd
from matplotlib import pyplot as plt
from ggplot import *
import time
import pickle

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 500)
pd.options.mode.chained_assignment = None  # default='warn'

#roll some dice and give me result
def roll(x,y=1,Dx=6,RR=0):
    roll = np.random.randint(1,Dx+1,x)
    if RR==1:
        roll[roll==1] = np.random.randint(1,Dx+1,roll[roll==1].shape[0])
    roll2 = roll[roll>=y]
    return roll2

#define to wound roll needed
def to_wound(S,T):
    if S>=2*T:
        return 2
    elif S>T:
        return 3
    elif S==T:
        return 4
    elif 2*S<=T:
        return 6
    else: 
        return 5

#define to save roll needed
def to_save(AP,SV,INV):
    if INV < (SV+AP):
        return INV
    else:
        return np.max([SV+AP,2])

#run single simulated series of shots from selected weapons to selected targets
def single_shot(CTH,MODELS,WEAPONS,SHOTS,S,T,SV,INV,AP,D,W,LD,RS=0,RD=0,FNP=7,ShRR=0,AutoHit=0,Melta=0,US=1):
    
    if RS==1:
        ATTACKS = roll(WEAPONS*MODELS, Dx=SHOTS).sum()
    else:
        ATTACKS = WEAPONS*SHOTS*MODELS
    
    hit = roll(ATTACKS,CTH,RR=ShRR).shape[0] * (1-AutoHit) + ATTACKS * AutoHit
    wound = roll(hit,to_wound(S,T)).shape[0]
    unsaved = wound-roll(wound,to_save(AP,SV,INV)).shape[0]
    
    total_wound = 0
    model_killed = 0
    battle_shock = 0
    wounds_remaining = W
    models_remaining = US
    
    if D>1:
        for x in range(1,unsaved+1):
            if models_remaining>0:
                if Melta==1:
                    Dmg = roll(2,Dx=D).max()
                else:
                    Dmg = roll(1, Dx=D)[0] * RD + D * (1-RD)
                    
                suffered = np.min([Dmg-roll(Dmg,FNP).shape[0],wounds_remaining])
                if suffered==wounds_remaining:
                    wounds_remaining=W
                    model_killed += 1
                    models_remaining -= 1
                else: 
                    wounds_remaining -= suffered
                
                total_wound = total_wound + suffered
    else:
        total_wound = unsaved-roll(unsaved,FNP).shape[0]
        total_wound = np.min([total_wound,US*W])
        model_killed = np.min([np.floor(total_wound/W),models_remaining])
        models_remaining=models_remaining-model_killed
    
    if model_killed>0:
        battle_shock = np.min([np.max([0,model_killed + roll(1)[0] - LD]),models_remaining])
    
    return np.array([total_wound,model_killed,battle_shock,US,models_remaining])

#import data from files
def import_join():
    enem=pd.read_csv('enemy.csv')
    weap=pd.read_csv('loadout.csv')
    shot=pd.read_csv('shooters.csv')
    cond=pd.read_csv('conditions.csv')
    
    enem['key'] = 0
    cond['key'] = 0
    
    joined=pd.merge(weap,shot, on='SH_KEY')
    joined['key'] = 0
    
    joined=pd.merge(joined,enem, on='key')
    joined=pd.merge(joined,cond[cond.CONDITION=='NO EFFECTS'], on='key')
    del joined['key']
    return joined

#run simulation given number of times based on input row from data frame
def sim_result(row,simulations=10000):
    sim = pd.DataFrame([single_shot(CTH=row['CTH'],MODELS=row['MODELS'],WEAPONS=row['WEAPONS'],SHOTS=row['SHOTS'],S=row['S'],T=row['T'],SV=row['SV'],INV=row['INV'],AP=row['AP'],D=row['D'],W=row['W'],LD=row['LD'],RS=row['RS'],RD=row['RD'],FNP=row['FNP'],ShRR=row['ShRR'],AutoHit=row['AutoHit'],Melta=row['Melta'],US=row['US']) for _ in range(simulations)])
    slice=0
    x=sim[slice].value_counts().sort_index()/sim.shape[0]
    x = pd.DataFrame(x)
    x=x.reindex(list(range(0,20)),fill_value=0)
    x.index += 1
    x=((1-x.cumsum())*100).transpose().round(1)
    names = list(map(str, range(1,21)))
    names = ["W_{0}".format(x) for x in names]
    x.columns = names
    x['GEAR']=row['GEAR']
    x['TARGET']=row['TARGET']
    x['SHOOTER']=row['SHOOTER']
    x['CONDITION']=row['CONDITION']
    x['SYSTEM']=row['SYSTEM']
    x['US'] = sim[3].max()
    print(row.name)
    return x.iloc[0,:]

#for testing
#int(test_df['US'])
#test_df['US']

#testing_shots = pd.DataFrame([single_shot(CTH=2,MODELS=1,WEAPONS=4,SHOTS=1,S=8,T=6,SV=3,INV=7,AP=4,D=6,W=8,LD=8,RS=0,RD=1,FNP=7,ShRR=0,AutoHit=0,Melta=1,US=1) for _ in range(10000)]); testing_shots
#testing_shots = pd.DataFrame([single_shot(CTH=int(test_df['CTH']),MODELS=int(test_df['MODELS']),WEAPONS=int(test_df['WEAPONS']),SHOTS=int(test_df['SHOTS']),S=int(test_df['S']),T=int(test_df['T']),SV=int(test_df['SV']),INV=int(test_df['INV']),AP=int(test_df['AP']),D=int(test_df['D']),W=int(test_df['W']),LD=int(test_df['LD']),RS=int(test_df['RS']),RD=int(test_df['RD']),FNP=int(test_df['FNP']),ShRR=int(test_df['ShRR']),AutoHit=int(test_df['AutoHit']),Melta=int(test_df['Melta']),US=int(test_df['US'])) for _ in range(10000)])
#testing_shots
#slice=0
#x=testing_shots[slice].value_counts().sort_index()/testing_shots.shape[0]
#x = pd.DataFrame(x)
#x=x.reindex(list(range(0,20)),fill_value=0)
#x.index += 1
#x=((1-x.cumsum())*100).transpose().round(1)
#names = list(map(str, range(1,21)))
#names = ["W_{0}".format(x) for x in names]
#x.columns = names    
#x.iloc[0,:]

#import data
df = import_join()
df

#for testing
#test_df = df[(df['SHOOTER'] == 'Commander') & (df['TARGET'] == '1x Drop Pod') & (df['GEAR'] == 'Fusion (melta)')]; test_df
#test_sim = test_df.apply(lambda row: sim_result(row), axis=1); test_sim


#run simulation on data 10000 times !!!!!!!!!!! TAKES MORE THAN 20 MINUTES
start = time.time()
multi_sim = df.apply(lambda row: sim_result(row), axis=1)
end = time.time()
print((end - start)/60)
pickle.dump(multi_sim, open("multi_sim.pickle", "wb"))
multi_sim = pickle.load(open("multi_sim.pickle", "rb"))

multi_sim
#define plot
def plot(unit):
    data=multi_sim[multi_sim['SHOOTER'] == unit]
    data=data.melt(id_vars=data.columns[20:26],value_name='Value', var_name='Wounds')
    data['Wounds']=data['Wounds'].str.replace('W_','').apply(int)
    data=data[data['Value']>=0.0001]
    data.sort_values(['TARGET','Wounds'],inplace=True)

    plot = ggplot(aes(x='Wounds', y='Value', color='SYSTEM'), data=data) +\
        geom_line() +\
        facet_grid('TARGET','GEAR') +\
        scale_x_continuous(name="Wounds", breaks=list(range(1,11)), limits=(1,10)) +\
        scale_y_continuous(name="% chance of scorring at least that many wounds", breaks=list(range(0,101,10))) +\
        scale_color_manual(values=['#FF0000','#00FF00','#0000FF','#000000']) +\
        ggtitle(('T\'au {} with different equipment').format(unit))
        #scale_color_manual(values=['#A6CEE3','#1F78B4','#B2DF8A','#33A02C','#FB9A99','#E31A1C','#FDBF6F','#FF7F00','#CAB2D6','#6A3D9A'])

    return plot



data=multi_sim[(multi_sim['SHOOTER']=='Commander')]
data=data.melt(id_vars=data.columns[20:26],value_name='Value', var_name='Wounds')
data['Wounds']=data['Wounds'].str.replace('W_','').apply(int)
data=data[data['Value']>=0.0001]
data.sort_values(['TARGET','Wounds'],inplace=True)
#data['GEAR_SYSTEM']=('{} + {}').format(data['GEAR'],data['SYSTEM'])
data['GEAR_SYSTEM']=data['GEAR'] + ' '  + data['SYSTEM']
data    

plot = ggplot(aes(x='Wounds', y='Value', color='GEAR'), data=data) +\
    geom_line() +\
    facet_grid('SYSTEM','TARGET') +\
    scale_color_manual(values=['#A6CEE3','#1F78B4','#B2DF8A','#33A02C','#FB9A99','#E31A1C','#FDBF6F','#FF7F00','#CAB2D6','#6A3D9A']) +\
    scale_x_continuous(name="Wounds", breaks=list(range(1,11)), limits=(1,10)) +\
    scale_y_continuous(name="% chance of scorring at least that many wounds", breaks=list(range(0,101,10)))

plot.save('commander_v2.jpg',width=25,height=25)

#save plot to file (and display it in plots window)
start = time.time()
plot('3xCrysis').save('crysis.jpg',width=40, height=40)
plot('Commander').save('commander.jpg',width=40, height=40)
end = time.time()
print((end - start)/60)





#sim = pd.DataFrame([single_shot(CTH=4,MODELS=3,WEAPONS=3,SHOTS=2,S=7,T=4,SV=2,INV=5,AP=1,D=3,W=2,LD=9,RS=0,RD=1,FNP=7,ShRR=1) for _ in range(10000)])
#slice=0
#plt.hist(sim[slice],bins=range(min(sim[slice]), max(sim[slice]) + 1, 1))
#x=sim[slice].value_counts().sort_index()/sim.shape[0]
#x.index += 1
#x = pd.DataFrame(x)
#x=x.reindex(list(range(1,21)),fill_value=0)
#x=((1-x.cumsum())*100).transpose().round(1)
#names = list(map(str, range(1,21)))
#names = ["W_{0}".format(x) for x in names]
#x.columns = names
#x

#names = list(map(str, range(1,21)))
#names = ["W_{0}".format(x) for x in names]
#names

#l = range(1,21)
#pd.DataFrame(data=[l])

#x=[]
#for _ in range(10000):
#    a=roll(4,2); b=roll(a.shape[0],2); x.append(b.shape[0]); x

#c=pd.DataFrame(x)[0].value_counts().sort_index()/pd.DataFrame(x).shape[0]
#1-c.cumsum()
#c


def plot(unit):
    data=multi_sim.query("SYSTEM != 'MT+ATS'")
    data=data.melt(id_vars=data.columns[20:26],value_name='Value', var_name='Wounds')
    data['Wounds']=data['Wounds'].str.replace('W_','').apply(int)
    data=data[data['Value']>=0.0001]
    data.sort_values(['TARGET','Wounds'],inplace=True)

    plot = ggplot(aes(x='Wounds', y='Value', color='SYSTEM', linetype='SHOOTER'), data=data) +\
        geom_line() +\
        facet_grid('TARGET','GEAR') +\
        scale_x_continuous(name="Wounds", breaks=list(range(1,11)), limits=(1,10)) +\
        scale_y_continuous(name="% chance of scorring at least that many wounds", breaks=list(range(0,101,10))) +\
        scale_color_manual(values=['#FF0000','#00FF00','#0000FF','#000000']) 
        #scale_color_manual(values=['#A6CEE3','#1F78B4','#B2DF8A','#33A02C','#FB9A99','#E31A1C','#FDBF6F','#FF7F00','#CAB2D6','#6A3D9A'])

    return plot

plot('3xCrysis').save('both.jpg',width=40, height=40)


def plot(unit):
    data=multi_sim.query("SYSTEM == 'Only Guns'").query("SHOOTER == '3xCrysis'")#.query("TARGET == '10x Space Marine'")
    data=data.melt(id_vars=data.columns[20:26],value_name='Value', var_name='Wounds')
    data['Wounds']=data['Wounds'].str.replace('W_','').apply(int)
    data=data[data['Value']>=0.0001]
    data.sort_values(['TARGET','Wounds'],inplace=True)

    plot = ggplot(aes(x='Wounds', y='Value', color='GEAR'), data=data) +\
        geom_line() +\
        facet_wrap('TARGET') +\
        scale_color_manual(values=['#000000','#cccccc','#8a2be2','#ff8c00','#228b22','#ffff00','#ff0000','#0000ff','#bebebe','#ff69b4']) +\
        scale_x_continuous(name="Wounds", breaks=list(range(1,11)), limits=(1,10)) +\
        scale_y_continuous(name="% chance of scorring at least that many wounds", breaks=list(range(0,101,10)))
        
    return plot

plot('3xCrysis').save('single3.jpg',width=20, height=20)


