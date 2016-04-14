#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy
from text_clustering import TextClustering
import helper_functions

__author__ = 'greg'

def merge_clusters_with_distinct_users(cluster1,cluster2,dist):
    """
    given two clusters - where each element in a tuple is a user id
    - we might have more such as the corresponding marking index which doesn't matter
    merge these clusters if and only if there are no common users between the clusters
    :param cluster1:
    :param cluster2:
    :param dist:
    :return:
    """
    cluster1_users = zip(*cluster1)[0]
    cluster2_users = zip(*cluster2)[0]
    overlap = [id_ for id_ in cluster1_users if id_ in cluster2_users]

    return overlap == []


def merge_close_clusters(threshold):
    def f(cluster1,cluster2,dist):
        return dist <= threshold

    return f


def agglomerative_merge(cluster_tree,Z,merge_func):
    for index1,index2,dist,_ in Z:
        index1 = int(index1)
        index2 = int(index2)

        child1 = cluster_tree[index1]
        child2 = cluster_tree[index2]

        if (child1 is None) or (child2 is None):
            cluster_tree.append(None)
        else:

            if merge_func(child1,child2,dist):
                combined_users = child1[:]
                combined_users.extend(child2)
                cluster_tree.append(combined_users)

                cluster_tree[index1] = None
                cluster_tree[index2] = None
            else:
                cluster_tree.append(None)

    return cluster_tree


class AnnotateClustering(TextClustering):
    def __init__(self,shape,project,param_dict):
        TextClustering.__init__(self,shape,project,param_dict)

    def __cluster__(self,markings,user_ids,tools,reduced_markings,image_dimensions,subject_id):
        unique_users = set(user_ids)
        if len(unique_users) <= 2:
            print("too few users")
            return [],0

        # to create the clusters, we need two things - the user id corresponding to each
        # transcription. And the index of the transcription (out of the list of all transcriptions)
        # Actually only need the user id to create the clusters, but the transcription index is needed
        # to do anything after that
        cluster_tree = []
        # pts is for the clustering
        pts = []
        # also check that we still have least 3 unique users after removing any problem transcriptions
        unique_users = set()
        for i,id_ in enumerate(user_ids):
            text = markings[i][4]
            # check to see if the text is empty
            # - either actually an empty transcription - not sure how likely that is, but better safe than sorry
            # - contains \n, in which case __set_special_characters__ returns ""
            # - is just some font tag stuff which is just ignored
            if self.__set_tags__(text) != "":
                cluster_tree.append([(id_,i),])
                pts.append(markings[i][:4])
                unique_users.add(id_)

        if len(unique_users) <= 2:
            return [],0

        # todo - play around with n_components, if e_v_r is than 0.95, maybe increase to 3?
        pca = PCA(n_components=2)
        pca.fit(pts)
        # print("explained variance is " + str(sum(pca.explained_variance_ratio_)))
        xy = pca.transform(pts)

        # do agglomerative clustering - right now based on ward distance
        Z = hierarchy.linkage(xy, 'ward')

        cluster_tree = agglomerative_merge(cluster_tree,Z,merge_clusters_with_distinct_users)

        clusters_retval = []
        for cluster in cluster_tree:
            # if None - this node is either above or below a "cap" node
            if cluster is None:
                continue
            if len(cluster) <= 2:
                continue

            # extract the markings for this specific cluster
            markings_in_cluster = [markings[index] for (_,index) in cluster]

            # now create the cluster
            next_cluster,percentage_agreement = self.__create_cluster__(markings_in_cluster)

            aggregate_text = next_cluster["center"][-1]
            # gap characters shouldn't really count towards overall consensus
            # so if the aggregate text is just gap characters or "uncertain" characters
            # treat this a bad cluster
            non_gap_disagreement_characters = len([c for c in aggregate_text if ord(c) not in [24,27]])

            if (percentage_agreement < 0.5) or (non_gap_disagreement_characters == 0):
                next_clusters,found_clusters = self.__additional_clustering__(markings_in_cluster)

                # out of the mess of clusters, we found some "better" ones
                # they may still have low consensus - but we'll let the experts figure it out
                if next_clusters != []:
                    clusters_retval.extend(next_clusters)
                elif found_clusters is False:
                    # we know we have a problem cluster, so again, let the experts figure it out
                    clusters_retval.append(next_cluster)

                # image_file = self.project.__image_setup__(subject_id)[0]
                # image = plt.imread(image_file)
                # fig, ax1 = plt.subplots(1, 1)
                # ax1.imshow(image)
                #
                # for x1,x2,y1,y2,_ in markings_in_cluster:
                #     plt.plot([x1,x2],[y1,y2])
                #
                # plt.show()
            else:
                clusters_retval.append(next_cluster)

        print("lines retired: " + str(len(clusters_retval)))
        return clusters_retval,0

    def __additional_clustering__(self,markings):
        """
        if we have a cluster with a low consensus - check what's happening
        are there transcriptions at a completely different angle? If so, put those
        into their own cluster. Any others will be treated like noise
        If no splitting happens - keep the problem cluster together - let the expert deal with it
        :param markings:
        :return:
        """
        # look for lines that have a really different angle
        # the ordering of the line coordinates is different for the hesse line reduction
        x1,x2,y1,y2,_,_ = zip(*markings)
        hesse_lines = helper_functions.hesse_line_reduction(zip(x1,y1,x2,y2))

        intercepts,angles = zip(*hesse_lines)
        angle_dist_matrix = np.zeros((len(markings),len(markings)))

        # the distance matrix is simply the distance between the line segments' angles
        for angle1 in range(len(markings)-1):
            for angle2 in range(angle1+1,len(markings)):
                distance = angles[angle1] - angles[angle2]
                angle_dist_matrix[angle1,angle2] = distance
                angle_dist_matrix[angle2,angle1] = distance

        Z = hierarchy.linkage(angle_dist_matrix, 'ward')
        cluster_tree = [[i,] for i in range(len(markings))]
        cluster_tree = agglomerative_merge(cluster_tree,Z,merge_close_clusters(0.25))

        # look for all clusters with at least two transcriptions in them
        # if we find any, apply levenshtein clustering on top of that
        second_pass_clusters = []

        for cluster in cluster_tree:
            # if None - this node is either above or below a "cap" node
            if cluster is None:
                continue
            # we just want at least 2 lines that are close together
            if len(cluster) >= 2:
                second_pass_clusters.append(cluster)

        if second_pass_clusters == []:
            # this is a problem set of transcriptions so create a cluster
            # and let the experts deal with it
            return [],False
        else:
            # only return aggregations for clusters with at least 3 transcriptions
            # if only 2 - don't report this as a problem cluster - hopefully with some more transcriptions
            # we will figure it out
            clusters_retval = []
            for cluster in second_pass_clusters:
                if len(cluster) >= 3:
                    markings_in_cluster = [markings[index] for index in cluster]

                    next_cluster = self.__create_cluster__(markings_in_cluster)
                    clusters_retval.append(next_cluster)

            return clusters_retval,True

    def __create_cluster__(self,markings_in_cluster):
        """
        given a list of markings in a single cluster, actually create that cluster
        i.e. aggregate the text, and create the cluster variable and add all the necessary info
        """
        # folger gives a variant listing which is the last element in the zip(*) command - irrelevant for annotate
        x1_values,x2_values,y1_values,y2_values,transcriptions,_ = zip(*markings_in_cluster)

        # first replace every tag (multi characters) with a single character token
        transcriptions = [self.__set_tags__(t) for t in transcriptions]

        # now deal with any characters with MAFFT has trouble with (i.e. replace " ", and non standard ascii characters)
        mafft_safe_trans = [self.__set_special_characters__(t) for t in transcriptions]

        # is everyone in agreement? - if so, don't bother even calling MAFFT
        # such external calls seem to be a problem on AWS so trying to minimize them
        # doing this now (as opposed to a few lines earlier) since I don't want capitalization differences
        # to affect things. The merge_aligned_text will still handle differences in capitalization but
        # we don't need to call MAFFT
        # note that if all the transcriptions are the same length we probably don't need to call MAFFT
        # but just to be safe, we will
        agreement = [mafft_safe_trans[i] == mafft_safe_trans[i+1] for i in range(len(mafft_safe_trans)-1)]

        # we don't have complete agreement (or near complete agreement) in the transcriptions
        if False in agreement:
            # call mafft to align these transcriptions
            aligned_transcriptions = self.__line_alignment__(mafft_safe_trans)
            # apply these alignments (insert gaps) to the original text with capitalization
            aligned_transcriptions = self.__add_alignment_spaces__(aligned_transcriptions,transcriptions)

            # in cases where there is disagreement, use voting to determine the most likely character
            # if there is strong disagreement, we'll mark those spots as unknown
            aggregate_text,percent_agreement= self.__merge_aligned_text__(aligned_transcriptions)

            # store the cluster members with alignment spaces added in
            dummy = [[] for _ in aligned_transcriptions]
            print(aligned_transcriptions)
            untokenized_text = [self.__reset_tags__(t) for t in aligned_transcriptions]
            print(untokenized_text)
            cluster_members = zip(x1_values,x2_values,y1_values,y2_values,untokenized_text,dummy)
        else:
            # we know that we have near complete agreement (up to disagreement about capitalizations) so we know
            # that the text is aligned
            aggregate_text,percent_agreement = self.__merge_aligned_text__(transcriptions)

            # we didn't do any alignment changes so the cluster members are the original pieces of text
            cluster_members = markings_in_cluster

        # convert tags from tokens back into multicharacter strings (so people can read them)
        aggregate_text = self.__reset_tags__(aggregate_text)

        # find the average start and end
        x1 = np.median(x1_values)
        x2 = np.median(x2_values)
        y1 = np.median(y1_values)
        y2 = np.median(y2_values)

        # cluster is what we'll actually return
        cluster = dict()
        cluster["center"] = (x1,x2,y1,y2,aggregate_text)

        cluster["tools"] = []

        cluster["cluster members"] = cluster_members

        cluster["num users"] = len(cluster["cluster members"])

        # collect some stats
        self.stats["retired lines"] += 1

        aggregate_text = cluster["center"][-1]

        errors = sum([1 for c in aggregate_text if ord(c) == 27])
        self.stats["errors"] += errors
        self.stats["characters"] += len(aggregate_text)

        return cluster,percent_agreement








# retired_subjects = [649216, 649217, 649218, 649219, 649220, 649222, 649223, 649224, 649225, 671754, 649227, 649228, 649229, 649230, 667309, 662872, 649234, 653332, 649239, 653336, 671769, 673482, 653339, 649244, 649245, 649246, 649248, 653346, 653348, 653349, 665642, 653356, 649261, 649262, 649263, 671752, 649267, 671796, 649269, 649270, 649272, 649226, 653316, 649281, 669763, 669765, 669766, 669769, 653390, 649231, 669793, 650598, 649233, 662887, 667764, 662890, 649362, 649363, 649365, 649366, 649368, 663705, 649370, 649371, 649372, 649373, 663710, 663711, 671904, 671905, 649378, 665627, 663717, 671910, 671911, 671912, 663722, 663723, 649388, 663726, 663727, 673480, 663730, 671923, 649397, 671927, 649400, 671932, 667338, 671936, 653344, 671938, 673483, 671940, 665634, 649426, 669923, 669926, 669927, 669932, 669933, 669937, 649461, 669942, 649463, 671992, 669946, 669947, 649258, 649470, 669951, 649472, 669954, 669955, 669957, 649478, 669960, 669962, 671447, 649484, 669965, 649486, 669969, 649490, 669972, 669976, 649497, 669978, 669981, 669982, 669984, 669985, 669987, 649508, 669989, 649510, 649265, 669992, 672896, 669995, 649505, 649518, 649519, 649520, 649521, 649522, 649523, 649524, 649525, 649526, 649527, 649528, 649529, 649530, 649531, 649532, 649533, 649534, 649535, 672283, 649537, 649538, 649540, 649541, 649542, 649543, 649544, 649545, 672081, 672083, 672084, 672086, 672087, 649564, 649573, 672102, 662929, 672104, 672106, 649592, 649601, 663939, 663940, 672136, 672834, 671811, 673518, 653381, 649639, 649644, 670127, 649653, 649578, 662945, 649647, 662948, 649511, 649728, 649744, 649745, 649747, 664089, 664090, 664091, 672284, 672285, 672286, 662961, 672297, 649780, 672310, 672313, 649788, 649793, 649798, 649828, 650342, 672360, 649836, 672366, 650344, 649845, 649848, 672385, 649858, 672409, 649890, 649892, 662641, 649900, 672541, 649907, 672439, 672443, 672885, 649922, 649937, 649942, 649944, 664301, 664303, 649369, 650366, 664313, 603264, 649995, 664334, 603268, 664346, 664349, 603270, 672551, 672554, 603271, 653341, 672558, 672561, 672562, 603273, 672906, 672907, 672567, 664396, 664403, 649387, 653114, 668518, 668522, 672915, 664454, 664463, 664469, 664471, 664473, 664474, 664488, 649374, 672714, 668628, 671909, 650212, 670700, 650688, 671914, 671915, 666640, 649390, 671919, 672797, 649462, 672803, 672808, 672809, 672810, 672814, 672815, 672817, 661465, 652344, 652347, 652354, 653010, 672858, 650336, 672870, 662632, 650346, 672875, 650349, 650352, 650353, 650355, 650357, 650358, 650360, 650361, 650363, 603262, 603263, 650368, 650369, 672898, 650371, 650372, 650374, 650375, 650377, 650378, 603275, 672909, 603279, 603280, 603281, 603283, 662676, 603285, 603288, 649466, 650400, 603297, 603299, 603300, 650411, 650413, 667334, 672952, 649608, 672968, 672973, 672977, 672980, 672295, 653015, 670979, 670993, 670998, 671007, 671014, 671017, 671019, 671022, 671024, 671030, 671034, 671036, 671037, 662847, 671042, 662851, 671047, 671048, 671061, 662870, 671064, 662874, 650590, 671073, 650594, 650596, 671078, 671079, 650602, 671083, 662894, 662895, 671088, 671089, 662898, 671091, 662901, 671094, 671099, 671101, 671102, 662922, 671115, 662924, 662926, 671119, 667024, 671121, 671125, 671126, 671128, 671129, 662941, 662943, 662944, 671137, 671138, 662947, 671140, 662949, 662950, 662951, 662953, 662955, 662956, 662958, 662959, 671153, 671154, 671156, 662910, 671158, 671161, 671163, 649203, 671168, 671170, 671173, 671174, 673481, 671179, 650701, 671184, 671640, 671189, 650636, 671994, 671198, 653364, 649467, 671204, 671206, 671208, 671209, 671220, 671222, 649471, 671230, 671232, 649474, 671249, 671253, 649477, 671267, 671269, 650797, 649480, 671282, 650809, 649482, 671296, 671302, 671303, 653239, 671330, 653240, 671327, 667234, 667238, 667240, 667242, 667244, 667248, 667252, 667258, 649496, 649498, 663204, 667303, 667305, 667306, 667307, 667308, 663213, 667313, 667315, 667318, 671417, 667322, 667324, 667326, 649504, 667331, 667332, 667333, 663238, 667335, 667336, 667337, 663242, 663243, 653004, 663245, 673486, 671442, 653011, 667350, 667351, 667352, 671449, 671452, 671453, 663262, 649509, 673505, 663268, 663269, 663270, 649213, 663278, 673520, 649512, 663282, 663283, 650537, 649514, 663296, 649515, 671939, 671510, 653273, 663323, 663324, 663326, 650544, 663330, 663334, 663335, 663337, 671532, 653307, 673594, 653279, 673606, 671565, 673614, 671568, 673620, 665435, 653149, 653151, 653152, 653154, 671588, 662844, 671594, 673672, 653310, 653294, 671638, 665495, 665496, 671727, 673693, 649200, 670022, 653231, 651185, 653234, 651187, 671671, 651192, 651193, 653243, 651197, 671678, 649205, 671680, 651201, 653250, 651203, 665540, 653253, 671686, 653257, 665547, 653260, 671693, 665550, 665551, 653304, 653270, 665559, 663512, 665561, 665564, 671709, 661471, 665569, 653283, 653285, 671718, 653288, 671721, 671722, 653291, 649198, 649199, 653296, 649201, 649202, 653299, 671733, 649206, 649207, 649208, 649209, 649211, 649212, 653309, 649214, 649215]
# import random
# random.seed(0)
# retired_subjects = retired_subjects[:200]



