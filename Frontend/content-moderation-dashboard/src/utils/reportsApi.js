// reportsApi.js - API calls for user reports

import axios from 'axios';
import EncryptionService from '../services/encryptionService';

const API_BASE_URL = 'http://192.168.1.13:4545/api/app'; // Update with your actual domain

// Mock response data
const MOCK_RESPONSE = {
  "isSuccess": true,
  "message": "success",
  "data": {
    "formattedReports": [
      {
        "id": 28,
        "postid": 428,
        "reportType": "Inappropriate Content",
        "comment": "",
        "email": "dhyannshah@gmail.com",
        "phone": "9429191918",
        "userid": 1,
        "createdAt": "2025-09-19T08:20:29.000Z",
        "updatedAt": "2025-09-19T08:20:29.000Z",
        "postDetails": {
          "title": "Nature is Best Therapy 🏔️🌿✨ \n #mountains  #nature  #peace  #trending  #whataview  #viral  #adventure  #explore #bigshorts",
          "coverfilename": "https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/coverFiles/84d7e491-4246-4817-9695-099c257cf8ca_COVER_IMAGE_mergedVideo_1720527346017.png",
          "videofilename": null,
          "post_type": "Post",
          "isinteractive": 0,
          "isforinteractivevideo": 1,
          "isforinteractiveimage": 0,
          "multipleposts": 0,
          "ispost": 1,
          "interactivevideo": "[{\"id\":0,\"parent_id\":-1,\"path\":\"https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/a95beae2-632b-4b1a-94f8-643bcaf70d9f_mergedVideo_1720527346017.mp4/hls/master.m3u8\",\"ios_streaming_url\":\"https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/a95beae2-632b-4b1a-94f8-643bcaf70d9f_mergedVideo_1720527346017.mp4/hls/master.m3u8\",\"android_streaming_url\":\"https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/a95beae2-632b-4b1a-94f8-643bcaf70d9f_mergedVideo_1720527346017.mp4/hls/master.m3u8\",\"duration\":\"9\",\"is_selcted\":false,\"on_video_end\":null,\"time_of_video_element_show\":null,\"audio_id\":2848,\"audio_file_path\":\"https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/audioFiles/f4fc5d9c-22c1-4675-9b07-623fe62721ec_audio_a3d7e76c-2d63-4478-afad-b854b8261629_1720527413337.mp3\",\"audio_name\":\"criccbuzzz's Original Audio\",\"audio_duration\":\"9.0\",\"functionality_datas\":null,\"post_id\":428,\"video_id\":640}]"
        }
      },

      {
        "id": 27,
        "postid": 2027,
        "reportType": "Content not Visible/ Playable",
        "comment": "",
        "email": "kshitizs2004@gmail.com",
        "phone": "878",
        "userid": 145,
        "createdAt": "2025-06-24T14:18:58.000Z",
        "updatedAt": "2025-06-24T14:18:58.000Z",
        "postDetails": {
          "title": "they hate each other",
          "coverfilename": "https://d198g8637lsfvs.cloudfront.net/Bigshorts/Flix/coverFiles/d819d16c-348c-4f7d-acf9-dbbaa7237fff_316214.png.webp",
          "videofilename": null,
          "post_type": "Post",
          "isinteractive": 0,
          "isforinteractivevideo": 0,
          "isforinteractiveimage": 1,
          "multipleposts": 0,
          "ispost": 1,
          "interactivevideo": "[{\"id\":0,\"parent_id\":-1,\"path\":\"https://d198g8637lsfvs.cloudfront.net/Bigshorts/Flix/InteractiveVideos/c71802af-bdbe-4d14-a214-98e3b890f6e1_316214.png.webp\",\"ios_streaming_url\":\"https://d198g8637lsfvs.cloudfront.net/Bigshorts/Flix/InteractiveVideos/c71802af-bdbe-4d14-a214-98e3b890f6e1_316214.png\",\"android_streaming_url\":\"https://d198g8637lsfvs.cloudfront.net/Bigshorts/Flix/InteractiveVideos/c71802af-bdbe-4d14-a214-98e3b890f6e1_316214.png\",\"duration\":\"5.0\",\"is_selcted\":false,\"on_video_end\":null,\"time_of_video_element_show\":\"/data/user/0/com.bigshorts.flutterapp.dev/cache/insta_assets_crop_0e1b5d46-e1c7-4855-94f4-eb6a9af87f918150239732110317486.jpg\",\"audio_id\":4508,\"audio_file_path\":\"https://d198g8637lsfvs.cloudfront.net/Bigshorts/Flix/audioFiles/426148b9-3f80-421f-87f7-15d06e54adc9_g_3953.m4a\",\"audio_name\":\"_1736583383476_egykcg_3953.m4a\",\"audio_duration\":\"15.0\",\"functionality_datas\":{\"list_of_buttons\":[],\"list_of_container_text\":[],\"list_of_images\":[],\"list_of_links\":[],\"list_of_locations\":[],\"list_of_polls\":[],\"music\":null,\"snip_share\":null,\"ssup_share\":null},\"aspect_ratio\":1.91,\"post_id\":2027,\"video_id\":2529,\"backdrop_gradient\":null}]"
        }
      },

      {
        "id": 24,
        "postid": 2025,
        "reportType": "Inappropriate Content",
        "comment": "test",
        "email": "kshitizs2004@gmail.com",
        "phone": "8780518358",
        "userid": 145,
        "createdAt": "2025-06-10T09:40:21.000Z",
        "updatedAt": "2025-06-10T09:40:21.000Z",
        "postDetails": {
          "title": "",
          "coverfilename": "https://d198g8637lsfvs.cloudfront.net/Bigshorts/Snip/coverFiles/f2186036-f536-46f2-ba6c-d073094d16e3_745341.jpg.webp",
          "videofilename": null,
          "post_type": "story",
          "isinteractive": 0,
          "isforinteractivevideo": 0,
          "isforinteractiveimage": 1,
          "multipleposts": 0,
          "ispost": 0,
          "interactivevideo": "[{\"id\":0,\"parent_id\":-1,\"path\":\"https://d198g8637lsfvs.cloudfront.net/Bigshorts/Snip/InteractiveVideos/fdc6622c-8428-4fc8-9427-836955eadae1_745341.jpg.webp\",\"ios_streaming_url\":\"https://d198g8637lsfvs.cloudfront.net/Bigshorts/Snip/InteractiveVideos/fdc6622c-8428-4fc8-9427-836955eadae1_745341.jpg\",\"android_streaming_url\":\"https://d198g8637lsfvs.cloudfront.net/Bigshorts/Snip/InteractiveVideos/fdc6622c-8428-4fc8-9427-836955eadae1_745341.jpg\",\"duration\":\"5.0\",\"is_selcted\":false,\"on_video_end\":null,\"time_of_video_element_show\":\"/storage/emulated/0/Pictures/1749119745341.jpg\",\"audio_id\":4506,\"audio_file_path\":\"https://d198g8637lsfvs.cloudfront.net/Bigshorts/Snip/audioFiles/3af56ce1-673d-469b-81d0-13cbd8ca0790_m_4499.m4a\",\"audio_name\":\"_1748590351495_nznipm_4499.m4a\",\"audio_duration\":\"7.0\",\"functionality_datas\":{\"list_of_buttons\":[],\"list_of_container_text\":[],\"list_of_images\":[],\"list_of_links\":[],\"list_of_locations\":[],\"list_of_polls\":[],\"music\":null,\"snip_share\":null,\"ssup_share\":null},\"aspect_ratio\":0,\"post_id\":2025,\"video_id\":2527,\"backdrop_gradient\":{\"colors\":[4288322715,4281414194,4284244063,4286019442,4289535775],\"begin\":{\"x\":-1,\"y\":-1},\"end\":{\"x\":1,\"y\":1}}}]"
        }
      },

      {
        "id": 22,
        "postid": 145,
        "reportType": "Other/ Content",
        "comment": "",
        "email": "",
        "phone": "",
        "userid": 1,
        "createdAt": "2025-04-26T11:12:47.000Z",
        "updatedAt": "2025-04-26T11:12:47.000Z",
        "postDetails": {
          "title": "\"Who's your favorite in the Mirzapur saga—Guddu Pandit or Munna Bhaiya? Dive into the chaos and let us know which character steals the show for you!\" \n #trending  #viral  #mirzapur  #action  #bollywood ",
          "coverfilename": "https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/coverFiles/27af1c46-49a9-47c8-82f3-b25c07848058_mergedImage_1719490362030.jpg",
          "videofilename": null,
          "post_type": "Post",
          "isinteractive": 1,
          "isforinteractivevideo": 1,
          "isforinteractiveimage": 0,
          "multipleposts": 0,
          "ispost": 1,
          "interactivevideo": "[{\"id\":0,\"parent_id\":-1,\"path\":\"https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/ce2c4fd4-6332-4f6a-9e73-2699164d1abd_mergedVideo_1719490153316.mp4/hls/master.m3u8\",\"duration\":\"15\",\"post_id\":145,\"video_id\":237},{\"id\":1,\"parent_id\":0,\"path\":\"https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/1db2010e-36af-43d2-840f-c5f94ed9c035_mergedVideo_1719490238676.mp4/hls/master.m3u8\",\"duration\":\"49\",\"post_id\":145,\"video_id\":238},{\"id\":2,\"parent_id\":0,\"path\":\"https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/1801d996-8d1d-4797-9a2f-28c2759c3931_mergedVideo_1719490324617.mp4/hls/master.m3u8\",\"duration\":\"40\",\"post_id\":145,\"video_id\":239}]"
        }
      },

      /** ---------------------------------------------------------
       *   ✅ YOUR NEW CASE INSERTED HERE AS `id: 21022`
       * ---------------------------------------------------------
       */
      {
        "id": 21022,
        "postid": 21022,
        "reportType": "Inappropriate Content",
        "comment": "",
        "email": "shjogani7@gmail.com",
        "phone": "",
        "userid": 389,
        "createdAt": "2025-11-01 20:32:00",
        "updatedAt": "2025-11-01 20:32:00",
        "postDetails": {
          "title": "Excellent question — and you’re super close to getting this right ✅",
          "coverfilename": "https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/coverFiles/90d11f69-c361-4a55-a0a7-614937d6357d_cover_thumbnail_318473.jpeg.webp",
          "videofilename": null,
          "post_type": "Post",
          "isinteractive": 1,
          "isforinteractivevideo": 1,
          "isforinteractiveimage": 0,
          "multipleposts": 0,
          "ispost": 1,
          "interactivevideo": "[{\"elementId\":\"\",\"functionality_datas\":{\"list_of_buttons\":[{\"id\":1762009292513,\"type\":2,\"radius\":3,\"buttonval\":5,\"color_for_txt_bg\":{\"background_color\":\"0xff00c4cc\",\"text_color\":\"0xff000000\"},\"on_action\":{\"video_path\":\"\",\"link_url\":null,\"id_of_video_list\":1,\"starting_time\":null,\"ending_time\":null,\"android_streaming_url\":null,\"ios_streaming_url\":null,\"skip_time_on_same_video\":null,\"linked_flix_id\":1818},\"background_shadow\":null,\"text_shadow\":null,\"border_color\":\"\",\"is_border\":true,\"text\":\"Type here\",\"is_selected\":false,\"text_family\":\"ComicNeue-Regular\",\"height\":3.3933070866141732,\"width\":27.983411838000844,\"top\":74.48444881889765,\"left\":22.22876652194446,\"starting_time\":0,\"ending_time\":4.72,\"album_model_id\":\"\",\"is_png\":9,\"noOfLines\":1,\"is_show\":null,\"last_next_video_jump_duration\":null,\"btn_alignment\":null,\"isForHotspot\":0,\"rotation\":0,\"font_size\":5.04}],\"list_of_container_text\":[],\"list_of_images\":[],\"list_of_links\":[],\"list_of_polls\":[]},\"current_time\":0,\"id\":0,\"parent_id\":-1,\"path\":\"https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/128d615f-e88c-4464-bf85-f30d9a169d56_311760.mp4/hls/master.m3u8\",\"duration\":\"4.72\",\"is_selcted\":false,\"video_id\":29557,\"android_streaming_url\":\"https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/128d615f-e88c-4464-bf85-f30d9a169d56_311760.mp4/hls/master.m3u8\",\"audio_duration\":\"30\",\"audio_file_path\":null,\"audio_id\":23066,\"postId\":0,\"audio_name\":\"usernames’s Original Audio\",\"time_of_video_element_show\":\"\",\"ios_streaming_url\":\"https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/128d615f-e88c-4464-bf85-f30d9a169d56_311760.mp4/hls/master.m3u8\",\"aspect_ratio\":1,\"post_id\":21022}]"
        }
      }
    ]
  }
};


/**
 * Fetch all reports with video URLs (with mock data for testing)
 * @param {boolean} useMock - If true, return mock data instead of API call
 * @returns {Promise} List of reports with post details and videos
 */
export const fetchReportsWithVideos = async (useMock = false) => {
  try {
    // For testing, return mock data
    if (useMock) {
      console.log('📦 Using mock data for testing');
      return new Promise((resolve) => {
        setTimeout(() => resolve(MOCK_RESPONSE)); // Simulate network delay
      });
    }

    // Real API call
    const token = localStorage.getItem('token');
    
    const response = await axios.get(`${API_BASE_URL}/getReportsWithVideos`, 
      {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error fetching reports with videos:', error);
    throw error;
  }
};


/**
 * Fetch MINI reports with video URLs
 * @param {boolean} useMock - If true, return mock data instead of API call
 * @returns {Promise} List of MINI reports with video details
 */
export const fetchMiniReportsWithVideos = async (useMock = false) => {
  try {
    // 🔹 MOCK DATA (if useMock = true)
    if (useMock) {
      console.log("🟢 MINI MOCK ACTIVE");
      return {
        isSuccess: true,
        data: {
          formattedReports: [
            {
              id: 999,
              postid: 21022,
              reportType: "Hateful or Abusive Content",
              comment: "User reported this mini video for review.",
              email: "test@example.com",
              phone: "9876543210",
              userid: 389,
              createdAt: new Date(),
              updatedAt: new Date(),
    
              flixDetails: {
                postId: 21022,
                postTitle:
                  "Excellent question — and you’re super close to getting this right ✅",
                languageId: -1,
                coverFile:
                  "https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/coverFiles/90d11f69-c361-4a55-a0a7-614937d6357d_cover_thumbnail_318473.jpeg.webp",
                userProfileImage: "",
                isAllowComment: 1,
                isPosted: 1,
                isCollab: 0,
                hasMultiplePosts: 0,
                nsfw: 0,
                tagUserCount: 0,
                scheduleTime: "2025-11-01 20:32:00",
                createdAt: "2025-11-01 20:32:00",
                userId: 389,
                userFullName: "Johan Doe",
                userName: "johandoe",
                userEmail: "shjogani7@gmail.com",
                isVerified: 0,
                likeCount: 1,
                superLikeCount: 0,
                dislikeCount: 0,
                saveCount: 0,
                commentCount: 3,
                isInteractive: "1",
    
                // ✅ FINAL Parsed interactiveVideo JSON (IMPORTANT)
                interactiveVideo: [
                  {
                    elementId: "",
                    id: 0,
                    parent_id: -1,
                    current_time: 0,
                    duration: "4.72",
                    is_selected: false,
                    video_id: 29557,
                    aspect_ratio: 1,
                    path: "https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/128d615f-e88c-4464-bf85-f30d9a169d56_311760.mp4/hls/master.m3u8",
                    android_streaming_url:
                      "https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/128d615f-e88c-4464-bf85-f30d9a169d56_311760.mp4/hls/master.m3u8",
                    ios_streaming_url:
                      "https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/128d615f-e88c-4464-bf85-f30d9a169d56_311760.mp4/hls/master.m3u8",
    
                    functionality_datas: {
                      list_of_buttons: [
                        {
                          id: 1762009292513,
                          type: 2,
                          radius: 3,
                          text: "Type here",
                          buttonval: 5,
                          color_for_txt_bg: {
                            background_color: "0xff00c4cc",
                            text_color: "0xff000000",
                          },
                          on_action: {
                            linked_flix_id: 1818, // ✅ This triggers the Redirect button
                            starting_time: null,
                            ending_time: null,
                          },
                          height: 3.39,
                          width: 27.98,
                          top: 74.48,
                          left: 22.23,
                          starting_time: 0,
                          ending_time: 4.72,
                          is_png: 9,
                          noOfLines: 1,
                          is_border: true,
                        },
                      ],
                      list_of_container_text: [],
                      list_of_images: [],
                      list_of_links: [],
                      list_of_polls: [],
                    },
                  },
                ],
    
                isForInteractiveImage: 0,
                isForInteractiveVideo: 1,
    
                videoFile: [
                  "https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/128d615f-e88c-4464-bf85-f30d9a169d56_311760.mp4/hls/master.m3u8",
                ],
    
                videoFile_base: [
                  "https://d1332u4stxguh3.cloudfront.net/Bigshorts/Flix/InteractiveVideos/128d615f-e88c-4464-bf85-f30d9a169d56_311760.mp4/hls/master.m3u8",
                ],
    
                latestCommentDetails: [
                  {
                    name: "Johan Doe",
                    username: "johandoe",
                    profileimage: "",
                    comment: "That’s",
                    userid: 389,
                    isFollow: 0,
                  },
                ],
    
                postTagUser: [],
                postLocation: [],
              },
            },
          ],
        },
      };
    }
    
    // 🔹 REAL API CALL
    const token = localStorage.getItem("token");

    const response = await axios.get(
      `${API_BASE_URL}/getMiniReportsWithVideos`,
      {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      }
    );

    return response.data;
  } catch (error) {
    console.error("Error fetching MINI reports:", error);
    throw error;
  }
};


/**
 * Update report status and post NSFW value
 * @param {string|number} reportId - Report ID (can include 'mini-' prefix)
 * @param {string} action - Action to take: 'reviewed', 'resolved', or 'rejected'
 * @param {string} sourceType - Source type: 'post' or 'mini'
 * @returns {Promise} Response from server
 */
export const updateReportStatus = async (reportId, action, sourceType) => {
  try {
    const token = localStorage.getItem('token');
    
    const response = await axios.post(
      `${API_BASE_URL}/updateReportStatus`,
      {
        reportId,
        action,
        sourceType
      },
      {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error updating report status:', error);
    throw error;
  }
};

// reportsApi.js

/**
 * Extract linked post IDs from Interactive SNIP functionality data
 * @param {Array} videos - Parsed interactive videos array
 * @returns {Array} Array of linked post IDs
 */
export const extractLinkedPosts = (videos) => {
  if (!videos || !Array.isArray(videos)) return [];

  const linkedPosts = [];

  videos.forEach(video => {
    // Check if functionality_datas exists
    if (video.functionality_datas && video.functionality_datas.list_of_buttons) {
      const buttons = video.functionality_datas.list_of_buttons;

      buttons.forEach(button => {
        // Check for linked_flix_id or linked_post_id in on_action
        if (button.on_action) {
          const linkedFlixId = button.on_action.linked_flix_id;
          const linkedPostId = button.on_action.linked_post_id;

          if (linkedFlixId) {
            linkedPosts.push({
              id: linkedFlixId,
              type: 'snip',
              source: 'linked_flix_id'
            });
          }

          if (linkedPostId) {
            linkedPosts.push({
              id: linkedPostId,
              type: 'snip',
              source: 'linked_post_id'
            });
          }
        }
      });
    }
  });

  return linkedPosts;
};


// Report type configurations
export const REPORT_TYPES = {
  INAPPROPRIATE: {
    label: 'Inappropriate Content',
    color: 'red',
    icon: '❌'
  },
  HATEFUL: {
    label: 'Hateful Or Abusive Content',
    color: 'red',
    icon: '❌'
  },
  COPYRIGHT: {
    label: 'Copyright/Trademark Infringe',
    color: 'orange',
    icon: '⚠️'
  },
  SPAM: {
    label: 'Spam or Misleading',
    color: 'yellow',
    icon: '⚠️'
  },
  TECHNICAL: {
    label: 'Content not Visible/Playable',
    color: 'blue',
    icon: '🔧'
  },
  OTHER: {
    label: 'Other/Content',
    color: 'gray',
    icon: '📝'
  }
};


export const STATUS_MAP = {
  0: 'pending',
  1: 'reviewed', 
  2: 'resolved',
  '-1': 'rejected'
};

export const STATUS_DISPLAY = {
  pending: { label: 'Pending', color: 'red', badge: 'bg-red-100 text-red-700' },
  reviewed: { label: 'Reviewed', color: 'yellow', badge: 'bg-yellow-100 text-yellow-700' },
  resolved: { label: 'Resolved', color: 'green', badge: 'bg-green-100 text-green-700' },
  rejected: { label: 'Rejected', color: 'gray', badge: 'bg-gray-100 text-gray-700' }
};

// Helper to convert numeric status to string
export const getStatusString = (numericStatus) => {
  return STATUS_MAP[numericStatus] || 'pending';
};

// Helper to get status display config
export const getStatusConfig = (status) => {
  const statusStr = typeof status === 'number' ? getStatusString(status) : status;
  return STATUS_DISPLAY[statusStr] || STATUS_DISPLAY.pending;
};

// Helper to get report type config
export const getReportTypeConfig = (reportType) => {
  const typeMap = {
    'Inappropriate Content': REPORT_TYPES.INAPPROPRIATE,
    'Hateful or Abusive Content': REPORT_TYPES.HATEFUL,
    'Copyright/Trademark Infringe': REPORT_TYPES.COPYRIGHT,
    'Spam or Misleading': REPORT_TYPES.SPAM,
    'Content not Visible/ Playable': REPORT_TYPES.TECHNICAL,
    'Other/ Content': REPORT_TYPES.OTHER
  };
  
  return typeMap[reportType] || REPORT_TYPES.OTHER;
};

// Helper to determine content type from post details
export const getContentTypeFromPost = (postDetails) => {
  console.log('🏷️ getContentTypeFromPost called with:', postDetails);
  
  if (!postDetails) return 'SNIP'; // Default
  
  const {
    ispost,
    isforinteractivevideo,
    isforinteractiveimage,
    isinteractive,
    post_type
  } = postDetails;
  
  console.log('🏷️ Checking conditions:', {
    ispost,
    isforinteractivevideo,
    isforinteractiveimage,
    isinteractive,
    post_type
  });
  
  // If isinteractive = 1 then Interactive SNIP (CHECK THIS FIRST!)
  if (isinteractive === 1) {
    console.log('✅ Detected: Interactive SNIP');
    return 'Interactive SNIP';
  }
  
  // If ispost = 1 and isforinteractivevideo = 1 and post_type = 'Post' then SNIP
  if (ispost === 1 && isforinteractivevideo === 1 && post_type === 'Post') {
    console.log('✅ Detected: SNIP');
    return 'SNIP';
  }
  
  // If post_type = 'story' then SSUP
  if (post_type === 'story') {
    console.log('✅ Detected: SSUP');
    return 'SSUP';
  }
  
  // If ispost = 1 and isforinteractiveimage = 1 and post_type = 'Post' then Shot
  if (ispost === 1 && isforinteractiveimage === 1 && post_type === 'Post') {
    console.log('✅ Detected: SHOT');
    return 'SHOT';
  }
  
  // Default fallback
  console.log('⚠️ Defaulting to: SNIP');
  return 'SNIP';
};


// Helper to parse interactive video JSON
// Helper to parse interactive video JSON and extract media based on content type
export const parseInteractiveVideos = (interactivevideo, contentType, multipleposts) => {
  console.log('🔍 parseInteractiveVideos called with:', {
    contentType,
    multipleposts,
    interactivevideoType: typeof interactivevideo,
    interactivevideoLength: interactivevideo?.length
  });

  if (!interactivevideo) return [];
  
  try {
    const parsed = typeof interactivevideo === 'string' 
      ? JSON.parse(interactivevideo) 
      : interactivevideo;
    
    console.log('📦 Parsed interactivevideo:', parsed);
    console.log('📦 Parsed is array:', Array.isArray(parsed));
    console.log('📦 Parsed length:', parsed?.length);
    
    // Handle array of media
    if (Array.isArray(parsed)) {
      
      // For Interactive SNIP, return all items (multiple videos with id: 0, 1, 2...)
      if (contentType === 'Interactive SNIP') {
        console.log('✅ Processing Interactive SNIP - returning all videos');
        const result = parsed.map(item => ({
          id: item.id,
          url: item.path || item.ios_streaming_url || item.android_streaming_url,
          type: getMediaType(item.path || item.ios_streaming_url || item.android_streaming_url),
          duration: item.duration,
          parent_id: item.parent_id,
          ...item
        }));
        console.log('🎬 Interactive SNIP result:', result);
        return result;
      }
      
      // For SHOT with multiple posts, return all items
      if (contentType === 'SHOT' && multipleposts === 1) {
        console.log('✅ Processing SHOT with multipleposts');
        return parsed.map(item => ({
          id: item.id,
          url: item.path || item.ios_streaming_url || item.android_streaming_url,
          type: getMediaType(item.path || item.ios_streaming_url || item.android_streaming_url),
          duration: item.duration,
          ...item
        }));
      }
      
      // For single media (SNIP, SHOT, SSUP), return only first item
      console.log('✅ Processing single media - returning first item only');
      if (parsed.length > 0) {
        const item = parsed[0];
        return [{
          id: item.id,
          url: item.path || item.ios_streaming_url || item.android_streaming_url,
          type: getMediaType(item.path || item.ios_streaming_url || item.android_streaming_url),
          duration: item.duration,
          ...item
        }];
      }
    }
    
    // Handle single object
    if (parsed.path || parsed.ios_streaming_url || parsed.android_streaming_url) {
      console.log('✅ Processing single object');
      return [{
        id: parsed.id || 0,
        url: parsed.path || parsed.ios_streaming_url || parsed.android_streaming_url,
        type: getMediaType(parsed.path || parsed.ios_streaming_url || parsed.android_streaming_url),
        duration: parsed.duration,
        ...parsed
      }];
    }
    
    console.log('⚠️ No valid data found');
    return [];
  } catch (error) {
    console.error('❌ Error parsing interactive videos:', error);
    return [];
  }
};

export const parseMiniInteractiveVideos = (interactivevideo) => {
  if (!interactivevideo) return [];
  
  try {
    const parsed = typeof interactivevideo === 'string' 
      ? JSON.parse(interactivevideo) 
      : interactivevideo;
    
    if (Array.isArray(parsed)) {
      return parsed.map(item => ({
        id: item.id,
        url: item.path || item.ios_streaming_url || item.android_streaming_url,
        type: getMediaType(item.path || item.ios_streaming_url || item.android_streaming_url),
        duration: item.duration,
        ...item
      }));
    }
    
    return [];
  } catch (error) {
    console.error('Error parsing MINI interactive videos:', error);
    return [];
  }
};

// Helper to determine media type from URL
export const getMediaType = (url) => {
  if (!url) return 'unknown';
  
  const urlLower = url.toLowerCase();
  
  // Check for video formats
  if (urlLower.includes('.m3u8')) return 'video';
  if (urlLower.endsWith('.mp4')) return 'video';
  if (urlLower.endsWith('.mov')) return 'video';
  
  // Check for image formats
  if (urlLower.endsWith('.png')) return 'image';
  if (urlLower.endsWith('.jpg')) return 'image';
  if (urlLower.endsWith('.jpeg')) return 'image';
  if (urlLower.endsWith('.webp')) return 'image';
  if (urlLower.endsWith('.gif')) return 'image';
  
  return 'unknown';
};