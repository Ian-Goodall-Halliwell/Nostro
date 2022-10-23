import os
from datetime import datetime

import shutil
import pytz
from data_pulling import degrunk_data
from qlib import init

from qlib.config import REG_CRY
import fire

from datetime import datetime
import pytz
from binance import Client
from data_pulling.degrunk_data import get_acct,clean_assets
import csv
from qlib import init
import model
from qlib.config import REG_CRY
from binance.helpers import round_step_size

import os
import fire
import threading
import queue

from auth import quickstart

import line_profiler

minimum = 50
#@profile
def create_orders():
    
    train = False
    
    
    d = datetime.now(pytz.utc)
    
    degrunk_data.append1m(d)
    provider_uri = "C:/Users/Ian/Documents/Financial testing/data_pulling/data-download/1m-qlib" 
    init(provider_uri={"1min":provider_uri}, region=REG_CRY)
    
    
    model.run(d,False)
    return d
def date_to_milliseconds(date_str):
    """Convert UTC date to milliseconds
    If using offset strings add "UTC" to date string e.g. "now UTC", "11 hours ago UTC"
    See dateparse docs for formats http://dateparser.readthedocs.io/en/latest/
    :param date_str: date in readable format, i.e. "January 01, 2018", "11 hours ago UTC", "now UTC"
    :type date_str: str
    """
    # get epoch value in UTC
    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=pytz.utc)
    # parse our date string
    d = date_str
    # if the date is not timezone aware apply UTC timezone
    if d.tzinfo is None or d.tzinfo.utcoffset(d) is None:
        d = d.replace(tzinfo=pytz.utc)

    # return the difference in time
    return int((d - epoch).total_seconds() * 1000.0)


#@profile
def doacct(client):
    snap = get_acct(client)
    #print(snap)
    snp = {}
    info = client.get_exchange_info()
    tempsnp = [str(sap['asset']) + "BUSD" for sap in snap if not str(sap['asset']) == 'BUSD']
    sell_out = {d['symbol'].split("BUSD")[0]:d['filters'][2]['stepSize'] for d in info['symbols'] if d['symbol'] in tempsnp}
    for itm in snap:
        
        
        if itm['asset'] == "USDT":
            continue
        if not itm["asset"] == "BUSD":
            if float(itm['free']) > float(sell_out[itm["asset"]]):
                snp[itm['asset']+"BUSD"] = {'amount':float(itm['free'])}
            else:
                continue
        else:
            snp[itm['asset']] = {'amount':float(itm['free'])}
    try:
        clean_assets(client)
    except Exception as e:
        print(e)
        pass
    
    return snp

#@profile  
def exect(d,default=True,):
    
    
    clist = {}
    with open("C:/Users/Ian/Documents/Financial testing/auth/clients.csv") as fl:
        reader = csv.reader(fl, delimiter=',')
        for row in reader:
            clist.update({row[0]:{'api_keys':[row[1],row[2]]}})
    with open("C:/Users/Ian/Documents/Financial testing/auth/spreadsheet_ids.csv") as ids:
        read = csv.reader(ids, delimiter=',')
        for rowf in read:
            clist[rowf[0]].update({'spreadsheet_ids':rowf[1]})
    q = queue.SimpleQueue()
    threads = []
    for r in clist:
        q.put({r:clist[r]})
    print(len(clist))
    #for _ in range(len(clist)):
    for _ in range(len(clist)):
        try:
            t = threading.Thread(target=threadexec,args=[q,default,d])
            t.start()
            threads.append(t)
        except Exception as e:
            print(e)

    # Wait all threads to finish.
    for t in threads:
        t.join()

def threadexec(q,default,d):
    args = q.get()
    o = list(args.keys())[0]
    items = args[o]
    print('Trading for {}'.format(o))
    

        # Call the Sheets API
    
    contb = quickstart.main(items['spreadsheet_ids'])[0]
    if contb == "TRUE":
        cont= True
    elif contb == "FALSE":
        cont=False
    else:
        return
    #result = service.spreadsheets().values().get(spreadsheetId=items['spreadsheet_ids'], range=['A2']).execute()
    
    client = Client(items['api_keys'][0],items['api_keys'][1])
    
    snp = doacct(client)
    if cont == False:
        cancel_all_orders(client)
        selll = {lins:snp[lins]['amount'] for lins in snp}
        info = client.get_exchange_info()
        
        sell_out = {d['symbol']:d['filters'][2]['stepSize'] for d in info['symbols'] if d['symbol'] in snp}
        for order3 in selll:
            if order3 == "BUSD":
                continue
            try:
                ordersiz = round_step_size(selll[order3],float(sell_out[order3]))
                print("sell {}, {}".format(order3,ordersiz))
                order = client.order_market_sell(
                    symbol=order3,
                    quantity=ordersiz)
                print("Completed {}".format(order3))
            except:
                try:
                    ordersiz = round_step_size(selll[order3],float(sell_out[order3])) - float(sell_out[order3])
                    ordersiz = round_step_size(ordersiz,float(sell_out[order3]))
                    print("sell {}, {}".format(order3,ordersiz))
                    order = client.order_market_sell(
                        symbol=order3,
                        quantity=ordersiz)
                    print("Completed {}".format(order3))
                except Exception as e:
                    print(e)
            
        return
    
    
    buy_order_list,sell_order_list = model.read_preds(snap=snp,clnam=o,time=d)
    
    if os.path.exists("C:/Users/Ian/Documents/Financial testing/order lists/order_list_{}.csv".format(o)):
        os.remove("C:/Users/Ian/Documents/Financial testing/order lists/order_list_{}.csv".format(o))
    with open("C:/Users/Ian/Documents/Financial testing/order lists/order_list_{}.csv".format(o), 'x',newline='') as file:
        writer = csv.writer(file)
        for row in buy_order_list:
            print(row)
            writer.writerow([row])
        writer.writerow('delim')
        for rown in sell_order_list:
            writer.writerow([rown])
    
    execute_orders(client=client,name=o)
#@profile
def cancel_all_orders(client,buyL=[],sellL=[]):
    op = client.get_open_orders()
    for trade in op:
        
        buyLi = [list(i.keys())[0] for i in buyL]
        sellLi = [list(i.keys())[0] for i in sellL]
        if trade['executedQty'] == '0.00000000':
            if float(trade['price'])*float(trade['origQty']) > minimum:
                if trade['symbol'] in buyLi:
                    if trade['side'] == 'BUY':
                        
                        buyL = [i for i in buyL if not (list(i.keys())[0] == trade['symbol'])]
                        continue
                    else:
                        result = client.cancel_order(symbol=trade['symbol'],
                                                orderId=trade['orderId'])
                        print(result)
                elif trade['symbol'] in sellLi:
                    if trade['side'] == 'SELL':
                        sellL = [i for i in sellL if not (list(i.keys())[0] == trade['symbol'])]
                        continue
                    else:
                        result = client.cancel_order(symbol=trade['symbol'],
                                                orderId=trade['orderId'])
                        print(result)
                else:
                    try:
                        result = client.cancel_order(symbol=trade['symbol'],
                                                    orderId=trade['orderId'])
                        print(result)
                    except:
                        pass
    return buyL, sellL

#@profile
def execute_orders(client,name): 
    if os.path.exists("model/failed_orders.csv"):
        os.remove("model/failed_orders.csv")
    info = client.get_exchange_info()
    
    with open("C:/Users/Ian/Documents/Financial testing/order lists/order_list_{}.csv".format(name), 'r') as filer:
        reader = csv.reader(filer)
        buylist = []
        selllist = []
        delm = 0
        for rowr in reader:
            if rowr == ['d', 'e', 'l', 'i', 'm']:
                delm = 1
                continue
            if delm == 0:
                buylist.append(eval(rowr[0]))
            if delm == 1:
                selllist.append(eval(rowr[0]))
                
    buylist, selllist =cancel_all_orders(client,buyL=buylist,sellL=selllist)
    
    buyl = [list(lin.keys())[0] for lin in buylist]
    selll = [list(lins.keys())[0] for lins in selllist]
    
    buy_out = {d['symbol']:d['filters'][2]['stepSize'] for d in info['symbols'] if d['symbol'] in buyl}
    buy_out_p = {d['symbol']:d['filters'][0]['tickSize'] for d in info['symbols'] if d['symbol'] in buyl}
    sell_out = {d['symbol']:d['filters'][2]['stepSize'] for d in info['symbols'] if d['symbol'] in selll}
        
    for sellstk in selllist:
        try:
            ordersiz = round_step_size(list(sellstk.values())[0],float(sell_out[list(sellstk.keys())[0]]))
            symbl = list(sellstk.keys())[0]
            print("sell {}, {}".format(list(sellstk.keys())[0],ordersiz))
            depth = client.get_order_book(symbol=symbl)
            order = client.order_limit_maker_sell(
                symbol=symbl,
                quantity=ordersiz,
                price=f"{float(depth['asks'][0][0]):.8f}"
            )
            print(order)
            print("Completed {}".format(sellstk))
        except:
            try:
                ordersiz = round_step_size(list(sellstk.values())[0],float(sell_out[list(sellstk.keys())[0]])) - float(sell_out[list(sellstk.keys())[0]])
                ordersiz = round_step_size(ordersiz,float(sell_out[list(sellstk.keys())[0]]))
                symbl = list(sellstk.keys())[0]
                print("sell {}, {}".format(list(sellstk.keys())[0],ordersiz))
                depth = client.get_order_book(symbol=symbl)
                order = client.order_limit_maker_sell(
                    symbol=symbl,
                    quantity=ordersiz,
                    price=f"{float(depth['asks'][0][0]):.8f}"
                )
                print(order)
                print("Completed {}".format(list(sellstk.keys())[0]))
            except Exception as e:
                print(e)
    
    for buystk in buylist:
        price = buystk[list(buystk.keys())[0]]['price']
        buystk = {list(buystk.keys())[0]:buystk[list(buystk.keys())[0]]['amnt']}
        try:
            ordersizb = round_step_size(list(buystk.values())[0],float(buy_out[list(buystk.keys())[0]]))
            print("buy {}, {}".format(list(buystk.keys())[0],ordersizb))
            symbl = list(buystk.keys())[0]
            depth = client.get_order_book(symbol=symbl)
            if price < float(depth['bids'][0][0]):
                
                price = round_step_size(price,float(buy_out_p[list(buystk.keys())[0]]))
                price = f"{float(price):.8f}"
            else:
                price = f"{float(depth['bids'][0][0]):.8f}"
                
            order = client.order_limit_maker_buy(
                symbol=symbl,
                quantity=ordersizb,
                price=price
                )
            print(order)
            print("Completed {}".format(buystk))
        except Exception as e:
            print(e)
            with open("model/failed_orders.csv",'a',newline='') as f:
                wr = csv.writer(f)
                wr.writerow([ordersizb,symbl])
       
def upd():
    client = Client("igEARWI7LNtjhzHa3zrNAMtLlLtUjnNb3VFHSHCf5Nlnga4h3vAzthAQKe8wLYlC",	"BM8EVK6TI5kHKQ7sORXpkwHet8mtq8alhOV5JJQ25kAIunKL7YkGgfc80inJad0I")
    orders = client.get_open_orders()
    orders = {d["symbol"]:d for d in orders}
    info = client.get_exchange_info()
    buy_out = {d['symbol']:d['filters'][2]['stepSize'] for d in info['symbols'] if d['symbol'] in orders}
    for order in orders.values():
        
        if order["side"] == "BUY":
            symbl = order['symbol']
            depth = client.get_order_book(symbol=symbl)
            lead_price = depth['bids'][0][0]
            if order['price'] != lead_price:
                lead_price_n = float(lead_price) - float(buy_out[order['symbol']])    
                quote = (float(order['price'])*float(order['origQty']) - float(order['executedQty'])*float(order['price']))/float(lead_price)
                
                ordersizb = round_step_size(quote,float(buy_out[order['symbol']]))
                
                print("updated {}, {}".format(order['symbol'],lead_price))
                symbl = order['symbol']
                result = client.cancel_order(symbol=order['symbol'],orderId=order['orderId'])
                try:
                    order = client.order_limit_maker_buy(
                        symbol=symbl,
                        quantity=ordersizb,
                        price=f"{float(lead_price):.8f}"
                        )
                except:
                    ordersizb = round_step_size(quote*0.9,float(buy_out[order['symbol']]))
                    order = client.order_limit_maker_buy(
                        symbol=symbl,
                        quantity=ordersizb,
                        price=f"{float(lead_price):.8f}"
                        )
        elif order["side"] == "SELL":
            symbl = order['symbol']
            depth = client.get_order_book(symbol=symbl)
            lead_price = depth['asks'][0][0]
            if order['price'] != lead_price:
                quote = (float(order['origQty']) - float(order['executedQty']))
                ordersizb = round_step_size(quote,float(buy_out[order['symbol']]))
                
                lead_price_n = float(lead_price) + float(buy_out[order['symbol']])
                print("updated {}, {}".format(order['symbol'],lead_price))
                result = client.cancel_order(symbol=order['symbol'],orderId=order['orderId'])
                order = client.order_limit_maker_sell(
                    symbol=symbl,
                    quantity=ordersizb,
                    price=f"{float(lead_price):.8f}"
                    )
   
        
                            
def mainfunc(d,execer=True,orders=True,update=False):
    if update == True:
        upd()
        return
        
    if os.path.exists("C:/Users/Ian/Documents/Financial testing/order lists"):
        shutil.rmtree("C:/Users/Ian/Documents/Financial testing/order lists")
        os.mkdir("C:/Users/Ian/Documents/Financial testing/order lists")
    if orders == True:
        d = create_orders()
    if d == False:
        return
    elif execer == True:
        pass
        exect(d)
if __name__ == '__main__':
    #mainfunc(datetime.now(pytz.utc))
    fire.Fire(mainfunc)
