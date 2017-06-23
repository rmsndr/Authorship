-- statewise
	select origin_state_abr,count(*) 
	from [dbo].[ontime_2013_final] 
	where cancelled = '0.0'
	and ISNULL(CAST(NULLIF(ltrim(rtrim(arr_delay_new)), 'NULL') AS numeric(10, 0)), 0) >15 
	group by origin_state_abr
	--and (origin_city_name = 'Dallas/Fort' or dest_city_name = 'Dallas/Fort'))
-- train table
create table ontime_train_2013_tab(
dep_time_blk varchar(50),
arr_time_blk varchar(50),
month numeric(10,0),
dated date,
WorkDay varchar(10),
unique_carrier varchar(50),
origin_city_name varchar(50),
dest_city_name varchar(50),
dephrs int,
arrhrs int,
schduration numeric(10,0),
distance numeric(10,0),
delay numeric(10,0)
)
insert into ontime_train_2013_tab select * from ontime_train_2013

--traindataextraction
create view ontime_train_2013 as
select	
	dep_time_blk ,
	arr_time_blk,
	cast(ontime_2013_final.month as numeric(10, 0)) as 'month' ,
	cast(fl_date  as date) as 'dated',
	iif((cast(fl_date as date)) in (select cast(date as date) from [dbo].[stock$] 
	where datepart(YEAR,cast(date as date))=2013 and unique_carrier='AA' and holidays=1),'Holiday',
	iif(cast(fl_date as date) in  (select value 
 from 
 ( select dateadd(day,2,cast(date as date)) 'a',dateadd(day,1,cast(date as date)) 'b', 
 dateadd(day,-2,cast(date as date)) 'c',dateadd(day,-1,cast(date as date)) 'd'
  from [DataScience].[dbo].[stock$] 
  where datepart(YEAR,cast(date as date))=2013 and unique_carrier='AA' and holidays=1) a
  unpivot
  (
  value
  for col in (a,b,c,d)
  ) un),'PreHoliday',(choose(datepart(dw,cast(fl_date as date)), 'WEEKEND','Weekday',
	'Weekday','Weekday','Weekday','WEEKEND','WEEKEND'))))  'WorkDay' ,
	unique_carrier,	
	origin_city_name, 	 	
	dest_city_name, 		
	convert(int,cast(crs_dep_time as numeric(10)))/ 100  as 'dephrs',		
	convert(int,cast(crs_arr_time as numeric(10))) / 100 as 'arrhrs' ,	
	cast(crs_elapsed_time as numeric(10, 0)) as 'schduration', 	
	cast (distance as numeric(10, 0)) 'distance' ,	
	ISNULL(CAST(NULLIF(ltrim(rtrim(arr_delay_new)), 'NULL') AS numeric(10, 0)), 0) as 'delay'
	--convert(int,arr_delay_new)  as 'Delay'
	from [dbo].[ontime_2013_final] 
	where cancelled = '0.0' and (origin_city_name = 'Dallas/Fort' or dest_city_name = 'Dallas/Fort')
-- from dfw
select 
	a.[month]
      ,a.[WorkDay]
      ,a.[unique_carrier]
      ,a.[dest_city_name]
      ,a.[dephrs]
      ,a.[arrhrs]
      ,a.[schduration]
	  ,b.temperature_f
	  ,b.visibility_mph
	  ,b.conditions
	  ,cast(a.[delay] as numeric(10,0)) as delay
	   from
(select * from [dbo].[ontime_train_2013_tab] where origin_city_name='Dallas/Fort') a
left outer join 
(select date,time_blk,temperature_f,visibility_mph,conditions from [dbo].[weather_data_05]
where cast(id as numeric(20,0)) in (select min(cast(id as numeric(20,0))) as id
from [dbo].[weather_data_05] 
where   city = 'Dallas' and datepart(year,cast(date as date)) = 2013 and airport_code='KDAL'
group by time_blk,date)) b
on a.dated=b.date and a.dep_time_blk=b.time_blk

-- Holiday
 select value
 from 
 ( select dateadd(day,2,cast(date as date)) 'a',dateadd(day,1,cast(date as date)) 'b', 
 dateadd(day,-2,cast(date as date)) 'c',dateadd(day,-1,cast(date as date)) 'd'
  from [DataScience].[dbo].[stock$] 
  where datepart(YEAR,cast(date as date))=2013 and unique_carrier='AA' and holidays=1) a
  unpivot
  (
  value
  for col in (a,b,c,d)
  ) un 
	