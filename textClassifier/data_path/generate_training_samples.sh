#############################################################
# automatically getting training samples day by day
# author: wangzejun
# date: 20181115
#############################################################

# generate news title / click pv / exposure pv / ctr table
startdate=`date -d "60 days ago" +%Y%m%d`
enddate=`date -d "0 days ago" +%Y%m%d`
hive -e "
drop table if exists tianyi_news_months;
create table tianyi_news_months as
select dt, newstype, uuid, title, clickpv, exppv, clickuv, clickpv/exppv as ctr from
(
select t2.dt, t4.newstype, t2.uuid, t5.title, t2.clickpv, t3.exppv, t2.clickuv from 
(
	select dt,
 		uuid,
		sum(clickpv) as clickpv,
		count(1) as clickuv
	from
	(
		select dt, 
			uuid,
			device_id,
			count(1) as clickpv
		from ods_cre_news_click
		where dt >= ${startdate} and dt < ${enddate} 
			group by dt, uuid, device_id
	) t1 group by dt, uuid
) t2
inner join
(
 	select dt,
 		uuid,
		count(1) as exppv
	from ods_cre_news_exposure
	where dt >= ${startdate} and dt < ${enddate} and pagetype <> '-'
	group by dt, uuid
) t3 on t2.dt = t3.dt and t2.uuid = t3.uuid
inner join 
(
 	select dt,
 		uuid,
		media_area as newstype
	from ods_article_features_tianyi
	where dt >= ${startdate} and dt < ${enddate}
			and from_unixtime(cast(TIMESTAMP AS INT), 'yyyyMMdd') = dt
	group by dt, media_area, uuid
) t4 on t2.dt = t4.dt and t2.uuid = t4.uuid
inner join 
(
	select dt,
		uuid,
		title
	from mds_cms_content_produce
	where dt >= ${startdate} and dt < ${enddate}
		and to_date(ftianyitime) = concat(substr(dt, 1, 4), '-', substr(dt, 5, 2), '-', substr(dt, 7, 2))
		and editorgroupid != 'test'
		and modelid != 'subject'
		and wap_url != '-'
		and online = '2'
		and toTianyi = '1'
		and unix_timestamp(rctime) - unix_timestamp(ctime) <= 604800
) t5 on t2.dt = t5.dt and t2.uuid = t5.uuid
) t6"


# get positive and negative samples
pv_pos_threshold=1000
ctr_pos_threshold=0.15
pv_neg_threshold=50
ctr_neg_threshold=0.005
hive -e "select dt, 
				newstype, 
				title, 
				clickpv, 
				exppv, 
				ctr
		from tianyi_news_months
		where clickpv >= ${pv_pos_threshold} and
			  ctr >= ${ctr_pos_threshold}
		" | awk -F'\t' '{print 1"\t"$3}' > train_samples
hive -e "select dt,
				newstype,
				title,
				clickpv,
				exppv,
				ctr
		from tianyi_news_months
		where clickpv <= ${pv_neg_threshold} and
			  ctr <= ${ctr_neg_threshold}
		" | awk -F'\t' '{print 0"\t"$3}' >> train_samples

