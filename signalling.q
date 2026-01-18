/ =========================================================
/ Formal Signalling Score (SIG) – kdb+/q Implementation
/ =========================================================

/ ---------- Utilities ----------

sideSign:{[side] $[side=`BUY; 1f; -1f]}

mid:{(x.bid + x.ask) % 2}

/ ---------- Intraday Volatility ----------

/ assumes mkt table exists
ret:
select
  symbol,
  time,
  log[mid x] - prev log[mid x]
from mkt

intradayVol:
select
  symbol,
  time,
  sqrt sum ret*ret
from ret
by symbol, time xbar 00:30:00

/ ---------- Active Drift ----------

activeDrift:{
  po:x;
  s:sideSign[po.side];

  m:select time, mid:mid x
    from mkt
    where symbol=po.symbol,
          time within (po.startTime; po.endTime);

  if[count m < 2; :0n];

  deltaP:last m.mid - first m.mid;

  vol:avg intradayVol[symbol=po.symbol,
                      time within (po.startTime; po.endTime)]`intradayVol;

  dur:(po.endTime - po.startTime) % 00:01:00;

  s * deltaP % (vol * sqrt dur)
}

/ ---------- Cancel Reversion ----------

reversionScore:{
  po:x;
  if[null po.cancelTime; :0n];

  s:sideSign[po.side];
  T0:po.cancelTime;
  tau:00:05:00;

  pre:select mid:mid x
      from mkt
      where symbol=po.symbol,
            time within (T0 - tau; T0);

  post:select mid:mid x
       from mkt
       where symbol=po.symbol,
             time within (T0; T0 + tau);

  if[count pre < 2 or count post < 2; :0n];

  dPre:last pre.mid - first pre.mid;
  dPost:last post.mid - first post.mid;

  R:s * (dPre - dPost);

  vol:avg intradayVol[symbol=po.symbol,
                      time within (T0 - tau; T0 + tau)]`intradayVol;

  R % (vol * sqrt 2 * tau)
}

/ ---------- Footprint Metrics ----------

timeAtBBO:{
  po:x;

  ce:select time, price
     from childEvent
     where orderId=po.orderId,
           eventType in (`NEW;`REPLACE);

  m:select time, bid, ask
    from mkt
    where symbol=po.symbol,
          time within (po.startTime; po.endTime);

  j:aj[`time; ce; m];

  tob:sum $[po.side=`BUY; j.price=j.bid; j.price=j.ask];

  tob % count j
}

replaceRate:{
  ce:select time
     from childEvent
     where orderId=x.orderId,
           eventType=`REPLACE;

  dur:(x.endTime - x.startTime) % 00:01:00;

  count ce % dur
}

sizeEntropy:{
  ce:select qty
     from childEvent
     where orderId=x.orderId,
           eventType in (`NEW;`REPLACE);

  p:ce.qty % sum ce.qty;

  - sum p * log p
}

/ ---------- Z-score Helper ----------

zscore:{(x - avg x) % dev x}

/ ---------- Final SIG ----------

SIG:{
  po:x;

  Za:activeDrift po;
  Zr:reversionScore po;

  fp:enlist
    timeAtBBO po,
    replaceRate po,
    sizeEntropy po;

  Zf:(zscore fp 0) + (zscore fp 1) - (zscore fp 2);

  0.4*Za + 0.4*Zr + 0.2*Zf
}

/ ---------- Report ----------

signallingReport:
select
  orderId,
  symbol,
  algo,
  notional,
  Z_active:activeDrift each parentOrder,
  Z_reversion:reversionScore each parentOrder,
  SIG:SIG each parentOrder
from parentOrder
